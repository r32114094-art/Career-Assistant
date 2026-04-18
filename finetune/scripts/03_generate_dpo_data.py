"""
03_generate_dpo_data.py — 生成 DPO 偏好对数据

策略:
  1. 用 SFT checkpoint 对验证集做推理，收集错误样本
  2. 错误样本 = rejected，人工/Teacher 修正后 = chosen
  3. 补充: Teacher 模型生成典型错误模式作为 rejected

用法:
  # 方式一: 从 SFT 模型错误中提取
  python scripts/03_generate_dpo_data.py --mode from_errors --sft-model checkpoints/sft_qwen25_3b --test-data data/sft/val.jsonl

  # 方式二: 用 Teacher 直接生成偏好对
  python scripts/03_generate_dpo_data.py --mode synthetic --count 400

  # 方式三: 两者合并
  python scripts/03_generate_dpo_data.py --mode both --sft-model checkpoints/sft_qwen25_3b --test-data data/sft/val.jsonl --count 200
"""

import json
import argparse
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES_PER_BATCH = 5

SCRIPT_DIR = Path(__file__).parent
FINETUNE_DIR = SCRIPT_DIR.parent

TEACHER_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEACHER_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TEACHER_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

with open(FINETUNE_DIR / "data" / "tools_schema.json", "r", encoding="utf-8") as f:
    TOOLS_SCHEMA_STR = json.dumps(json.load(f), ensure_ascii=False, indent=2)


# ── 方式一: 从 SFT 模型错误中提取 ────────────────────

def collect_sft_errors(model_path: str, test_data_path: str) -> list[dict]:
    """用 SFT 模型推理验证集，收集与 ground truth 不一致的样本。"""
    import sys
    sys.path.insert(0, str(FINETUNE_DIR / "eval"))
    from reward_function import tool_call_reward, parse_tool_call

    # 加载测试数据
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # 推理
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"📦 加载 SFT 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    error_pairs = []
    for i, item in enumerate(test_data):
        messages = item["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gt_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if not gt_msg:
            continue

        # 推理
        text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)

        # 对比
        chosen = gt_msg["content"]
        if prediction.strip() != chosen.strip():
            user_msg = next((m["content"] for m in prompt_msgs if m["role"] == "user"), "")
            error_pairs.append({
                "prompt": user_msg,
                "chosen": chosen,
                "rejected": prediction,
                "source": "sft_error",
            })

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(test_data)}, 已收集 {len(error_pairs)} 个错误对")

    print(f"✅ 从 SFT 模型错误中收集到 {len(error_pairs)} 个偏好对")
    return error_pairs


# ── 方式二: Teacher 生成合成偏好对 ─────────────────────

DPO_CATEGORIES = {
    "tool_confusion": {
        "count": 100,
        "instruction": """生成"工具选择混淆"的偏好对。
chosen: 正确的工具调用
rejected: 选了一个容易混淆的错误工具
例如: 用户说"面试题" → chosen=get_interview_questions, rejected=ask_ai_question""",
    },
    "param_hallucination": {
        "count": 80,
        "instruction": """生成"参数幻觉"的偏好对。
chosen: 只提取用户明确说的参数
rejected: 幻觉出用户未提供的参数值
例如: 用户说"搜岗位" → chosen=追问 role/location, rejected=search_jobs(role="AI工程师",location="北京") 其中 role 和 location 都是猜的""",
    },
    "false_positive": {
        "count": 80,
        "instruction": """生成"不该调用但调用了"的偏好对。
chosen: 友好拒绝（超范围请求）
rejected: 强行匹配某个工具进行调用
例如: 用户说"帮我翻译这段英文" → chosen=友好拒绝, rejected=ask_ai_question(question="翻译...")""",
    },
    "format_error": {
        "count": 60,
        "instruction": """生成"格式错误"的偏好对。
chosen: 标准的 <tool_call>JSON</tool_call> 格式
rejected: 格式错误，如缺闭合标签、多余文本混入、JSON 语法错误等
例如: rejected 可能是 tool_call 后面还跟了一段解释文本""",
    },
    "ask_vs_guess": {
        "count": 80,
        "instruction": """生成"追问 vs 猜测"的偏好对。
chosen: 参数不足时礼貌追问
rejected: 猜测缺失参数后直接调用工具
例如: 用户说"搜工作" → chosen=追问角色和地点, rejected=search_jobs(role="AI工程师", location="北京") 全是猜的""",
    },
}


def generate_synthetic_dpo(client: OpenAI, category: str, batch_size: int = 20) -> list[dict]:
    """用 Teacher 生成合成偏好对。"""
    cat_info = DPO_CATEGORIES[category]
    prompt = f"""你是训练数据生成专家。请生成 DPO (Direct Preference Optimization) 偏好对数据。

## 工具定义
{TOOLS_SCHEMA_STR}

## 类别: {category}
{cat_info['instruction']}

## 格式要求
生成 {batch_size} 条，每条格式:
{{"prompt": "用户请求", "chosen": "正确输出", "rejected": "错误输出"}}

- chosen 中如果是工具调用，必须用 <tool_call>{{"name":"...","arguments":{{...}}}}</tool_call> 格式
- rejected 中包含上述说明的特定错误类型
- 输出为 JSON 数组

请开始生成:"""

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": "你是 DPO 训练数据生成引擎。只输出合法 JSON 数组。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=8192,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
        for item in data:
            item["source"] = f"synthetic_{category}"
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"  ⚠️ 生成失败: {e}")
        return []


# ── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 DPO 偏好对数据")
    parser.add_argument("--mode", choices=["from_errors", "synthetic", "both"], default="synthetic")
    parser.add_argument("--sft-model", type=str, help="SFT 模型路径 (from_errors/both 模式)")
    parser.add_argument("--test-data", type=str, help="验证集路径 (from_errors/both 模式)")
    parser.add_argument("--count", type=int, default=None, help="合成数据覆盖各类别总数")
    parser.add_argument("--output", type=str, default="data/dpo/preferences.jsonl")
    args = parser.parse_args()

    all_pairs = []

    # 方式一: 从 SFT 错误中提取
    if args.mode in ("from_errors", "both"):
        if not args.sft_model or not args.test_data:
            print("❌ from_errors 模式需要 --sft-model 和 --test-data")
            return
        error_pairs = collect_sft_errors(args.sft_model, args.test_data)
        all_pairs.extend(error_pairs)

    # 方式二: Teacher 合成
    if args.mode in ("synthetic", "both"):
        if not TEACHER_API_KEY:
            print("❌ 未设置 DEEPSEEK_API_KEY 环境变量。请在 .env 文件中配置。")
            return
        client = OpenAI(api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL)
        for category, info in DPO_CATEGORIES.items():
            target = args.count // len(DPO_CATEGORIES) if args.count else info["count"]
            print(f"\n📝 生成 DPO 类别: {category} (目标: {target} 条)")

            collected = []
            consecutive_failures = 0
            while len(collected) < target:
                batch_size = min(20, target - len(collected))
                batch = generate_synthetic_dpo(client, category, batch_size)
                if batch:
                    collected.extend(batch)
                    consecutive_failures = 0
                    print(f"  累计: {len(collected)}/{target}")
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_RETRIES_PER_BATCH:
                        print(f"  ❌ 连续 {MAX_RETRIES_PER_BATCH} 次失败，跳过")
                        break
                    print(f"  ⚠️ 失败 ({consecutive_failures}/{MAX_RETRIES_PER_BATCH})")
                time.sleep(1)

            all_pairs.extend(collected[:target])

    # 保存
    output_path = FINETUNE_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✅ 共生成 {len(all_pairs)} 个 DPO 偏好对 → {output_path}")


if __name__ == "__main__":
    main()
