"""
evaluate.py — 统一评估脚本

对任意模型（微调模型 / DeepSeek 基线）运行标准化评估，输出 6 个核心指标。

用法:
  # 评估本地微调模型
  python eval/evaluate.py --model path/to/checkpoint --test-data data/sft/test.jsonl

  # 评估 DeepSeek-V3 基线 (zero-shot)
  python eval/evaluate.py --api deepseek --test-data data/sft/test.jsonl

  # 评估并保存结果
  python eval/evaluate.py --model path/to/checkpoint --test-data data/sft/test.jsonl --output eval/results/e1_sft_only.json
"""

import json
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# 确保 eval/ 目录在导入路径中
EVAL_DIR = Path(__file__).parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


def load_test_data(test_path: str) -> list[dict]:
    """加载测试数据。每条包含 messages (含 system+user) 和 metadata (含 ground truth)。"""
    data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def extract_prompt_and_gt(item: dict) -> tuple[list[dict], dict]:
    """从测试样本中分离出 prompt (不含 assistant 回复) 和 ground_truth。"""
    messages = item["messages"]
    # prompt = 除最后一条 assistant 消息外的所有消息
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    # ground truth 从 metadata 中提取
    metadata = item.get("metadata", {})
    raw = metadata.get("raw", {})

    if "tool_call" in raw:
        gt = {
            "action": "call",
            "name": raw["tool_call"]["name"],
            "required_params": raw["tool_call"].get("arguments", {}),
            "optional_params": {},
        }
    elif metadata.get("category") == "reject":
        gt = {"action": "reject"}
    elif metadata.get("category") == "ask_followup":
        gt = {"action": "ask_followup"}
    elif "tool_call_or_response" in raw:
        # multi_turn 类别：tool_call_or_response 可能是工具调用或文本
        resp = raw["tool_call_or_response"]
        if isinstance(resp, str) and "<tool_call>" in resp:
            gt = {"action": "call", "name": "", "required_params": {}}
        else:
            gt = {"action": "reject"}
    else:
        # 兜底：从 assistant 回复推断
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_msg and "<tool_call>" not in assistant_msg.get("content", ""):
            gt = {"action": "reject"}
        else:
            gt = {"action": "call", "name": "", "required_params": {}}

    return prompt_messages, gt


def run_local_inference(model_path: str, prompts: list[list[dict]]) -> tuple:
    """用本地模型跑推理。

    Returns:
        (predictions: list[str], model, tokenizer) — 返回模型对象以便延迟测试复用。
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"📦 加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        predictions = []
        for i, messages in enumerate(prompts):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(pred)

            if (i + 1) % 10 == 0:
                print(f"  推理进度: {i+1}/{len(prompts)}")

        return predictions, model, tokenizer

    except ImportError:
        print("⚠️ 未安装 transformers，请安装: pip install transformers torch")
        return [], None, None


def run_api_inference(api_type: str, prompts: list[list[dict]]) -> list[str]:
    """用 API 模型跑推理（DeepSeek / OpenAI）。"""
    import os
    from openai import OpenAI

    if api_type == "deepseek":
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        raise ValueError(f"不支持的 API 类型: {api_type}")

    predictions = []
    for i, messages in enumerate(prompts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            pred = response.choices[0].message.content
            predictions.append(pred)
        except Exception as e:
            print(f"  ⚠️ API 调用失败 (样本 {i}): {e}")
            predictions.append("")

        if (i + 1) % 10 == 0:
            print(f"  推理进度: {i+1}/{len(prompts)}")
        time.sleep(0.5)  # 限速

    return predictions


def measure_latency(
    model, tokenizer, sample_prompt: list[dict], n_runs: int = 20
) -> dict:
    """测量推理延迟 (p50/p95)。复用已加载的模型，避免重复加载。"""
    try:
        import torch
        import numpy as np

        text = tokenizer.apply_chat_template(sample_prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 预热 3 次（GPU 频率稳定）
        for _ in range(3):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "mean_ms": round(float(np.mean(latencies)), 1),
            "n_runs": n_runs,
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Function Call 模型评估")
    parser.add_argument("--model", type=str, help="本地模型路径")
    parser.add_argument("--api", type=str, choices=["deepseek"], help="API 模型类型")
    parser.add_argument("--test-data", type=str, required=True, help="测试数据路径 (JSONL)")
    parser.add_argument("--output", type=str, help="结果保存路径")
    parser.add_argument("--latency", action="store_true", help="是否测量延迟")
    args = parser.parse_args()

    # 加载测试数据
    print("📂 加载测试数据...")
    test_data = load_test_data(args.test_data)
    print(f"  共 {len(test_data)} 条测试样本")

    # 分离 prompt 和 ground truth
    prompts = []
    ground_truths = []
    for item in test_data:
        prompt, gt = extract_prompt_and_gt(item)
        prompts.append(prompt)
        ground_truths.append(gt)

    # 推理
    model_name = args.model or f"api:{args.api}"
    print(f"\n🚀 开始推理: {model_name}")

    _model_obj = None
    _tokenizer_obj = None
    if args.model:
        predictions, _model_obj, _tokenizer_obj = run_local_inference(args.model, prompts)
    elif args.api:
        predictions = run_api_inference(args.api, prompts)
    else:
        print("❌ 请指定 --model 或 --api")
        return

    # 评估
    print("\n📊 评估中...")
    from reward_function import evaluate_batch
    results = evaluate_batch(predictions, ground_truths)

    # 延迟测试
    if args.latency and args.model and _model_obj is not None:
        print("\n⏱️ 测量延迟...")
        latency = measure_latency(_model_obj, _tokenizer_obj, prompts[0])
        results["latency"] = latency

    # 打印结果
    print("\n" + "=" * 60)
    print(f"评估结果 — {model_name}")
    print("=" * 60)
    print(f"  工具选择准确率:   {results['tool_selection_accuracy']}")
    print(f"  参数提取 F1:      {results['param_extraction_f1']}")
    print(f"  格式合规率:       {results['format_compliance_rate']}")
    print(f"  拒绝准确率:       {results['rejection_accuracy']}")
    print(f"  误报率:           {results['false_positive_rate']}")
    print(f"  平均奖励分:       {results['avg_reward']}")
    if "latency" in results:
        lat = results["latency"]
        print(f"  推理延迟 p50:     {lat.get('p50_ms', 'N/A')} ms")
        print(f"  推理延迟 p95:     {lat.get('p95_ms', 'N/A')} ms")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 移除 per_sample 详情以减小文件体积（可选保留）
        save_results = {k: v for k, v in results.items() if k != "per_sample"}
        save_results["model"] = model_name
        save_results["test_data"] = args.test_data
        save_results["test_samples"] = len(test_data)
        save_results["timestamp"] = datetime.now(timezone.utc).isoformat()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
