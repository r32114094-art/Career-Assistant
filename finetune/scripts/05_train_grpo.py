"""
05_train_grpo.py — GRPO 训练脚本

在 DPO checkpoint 基础上，用自定义奖励函数做 Group Relative Policy Optimization。

核心原理:
  1. 每个 prompt 生成 G 个候选回复 (group)
  2. 奖励函数对每个候选打分
  3. 组内相对排序：比组内平均好的 → 增强，比平均差的 → 抑制
  4. 无需人工标注偏好对，只需奖励函数

用法:
  python scripts/05_train_grpo.py --config configs/grpo_config.yaml
"""

import json
import yaml
import argparse
import sys
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    args = parser.parse_args()

    finetune_dir = Path(__file__).parent.parent
    config = load_config(finetune_dir / args.config)

    # 导入奖励函数
    sys.path.insert(0, str(finetune_dir / "eval"))
    from reward_function import tool_call_reward

    # ── 1. 加载 DPO checkpoint ──
    from unsloth import FastLanguageModel

    model_name = str(finetune_dir / config["model"]["name"])
    max_seq_length = config["model"]["max_seq_length"]

    print(f"📦 加载 DPO 模型: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=(config["lora"]["quantization"] == "4bit"),
    )

    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── 2. 加载 GRPO prompt 数据 ──
    grpo_cfg = config["grpo"]
    data_path = finetune_dir / config["data"]["prompts_path"]
    print(f"📂 加载 GRPO prompts: {data_path}")

    prompts_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts_data.append(json.loads(line.strip()))

    print(f"  共 {len(prompts_data)} 条 prompt")

    # 提取 prompt (message list) 和 ground truth
    # ⚠️ GRPOTrainer 内部会自动调 apply_chat_template，所以 prompt 列存 raw messages
    from datasets import Dataset

    def extract_prompt_messages(item):
        """从 messages 列表中提取 prompt 消息（不含 assistant 回复）。"""
        messages = item.get("messages", [])
        return [m for m in messages if m["role"] != "assistant"]

    prompt_messages = [extract_prompt_messages(item) for item in prompts_data]
    ground_truths = []
    for item in prompts_data:
        raw = item.get("metadata", {}).get("raw", {})
        if "tool_call" in raw:
            ground_truths.append({
                "action": "call",
                "name": raw["tool_call"]["name"],
                "required_params": raw["tool_call"].get("arguments", {}),
            })
        elif "tool_call_or_response" in raw:
            # multi_turn 类别：tool_call_or_response 可能是工具调用或文本
            resp = raw["tool_call_or_response"]
            if isinstance(resp, str) and "<tool_call>" in resp:
                ground_truths.append({"action": "call", "name": "", "required_params": {}})
            else:
                ground_truths.append({"action": "reject"})
        else:
            category = item.get("metadata", {}).get("category", "")
            if category == "ask_followup":
                ground_truths.append({"action": "ask_followup"})
            else:
                ground_truths.append({"action": "reject"})

    # 构建 dataset，ground_truth 列会通过 **kwargs 自动传递给 reward_fn
    dataset = Dataset.from_dict({
        "prompt": prompt_messages,
        "ground_truth": [json.dumps(gt, ensure_ascii=False) for gt in ground_truths],
    })

    # ── 3. 定义奖励函数包装器 ──
    # TRL GRPOTrainer:
    #   - completions 格式: list[list[dict]]，每个 completion = [{"role":"assistant","content":"..."}]
    #   - dataset 额外列通过 **kwargs 自动传入

    def reward_fn(completions, ground_truth=None, **kwargs) -> list[float]:
        """TRL GRPOTrainer 兼容的奖励函数。

        Args:
            completions: 模型生成的回复，list[list[dict]] 或 list[str]
            ground_truth: 对应的标准答案 JSON 字符串列表（从 dataset 列自动注入）

        Returns:
            list[float]: 对应的奖励分数
        """
        rewards = []
        for i, completion in enumerate(completions):
            # 提取文本: list[dict] 格式 → completion[0]["content"]
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                text = completion[0].get("content", "")
            elif isinstance(completion, str):
                text = completion
            else:
                text = str(completion)

            # 获取 ground truth
            if ground_truth and i < len(ground_truth):
                gt = json.loads(ground_truth[i]) if isinstance(ground_truth[i], str) else ground_truth[i]
            else:
                gt = {"action": "reject"}

            result = tool_call_reward(text, gt)
            rewards.append(result["total"])

        return rewards

    # ── 4. GRPO 训练 ──
    from trl import GRPOTrainer, GRPOConfig

    train_cfg = config["training"]
    output_dir = str(finetune_dir / train_cfg["output_dir"])

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        seed=config.get("seed", 42),
        report_to=config.get("report_to", "none"),
        # GRPO 特有参数
        num_generations=grpo_cfg["group_size"],
        max_new_tokens=grpo_cfg["max_new_tokens"],
        temperature=grpo_cfg["temperature"],
        beta=grpo_cfg["kl_penalty"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=grpo_config,
    )

    print(f"\n🚀 开始 GRPO 训练...")
    print(f"  Group size: {grpo_cfg['group_size']}")
    print(f"  Temperature: {grpo_cfg['temperature']}")
    print(f"  KL penalty: {grpo_cfg['kl_penalty']}")
    trainer.train()

    # ── 5. 保存 ──
    print(f"\n💾 保存模型: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    info = {
        "base_model": model_name,
        "stage": "grpo",
        "group_size": grpo_cfg["group_size"],
        "kl_penalty": grpo_cfg["kl_penalty"],
        "train_prompts": len(prompts_data),
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("✅ GRPO 训练完成!")


if __name__ == "__main__":
    main()
