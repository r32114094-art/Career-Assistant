"""
04_train_dpo.py — DPO 训练脚本

在 SFT checkpoint 基础上，用偏好对数据做 Direct Preference Optimization。

用法:
  python scripts/04_train_dpo.py --config configs/dpo_config.yaml
"""

import json
import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DPO 训练")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    args = parser.parse_args()

    finetune_dir = Path(__file__).parent.parent
    config = load_config(finetune_dir / args.config)

    # ── 1. 加载 SFT checkpoint ──
    from unsloth import FastLanguageModel

    model_name = str(finetune_dir / config["model"]["name"])
    max_seq_length = config["model"]["max_seq_length"]

    print(f"📦 加载 SFT 模型: {model_name}")
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

    # ── 2. 加载 DPO 数据 ──
    data_path = finetune_dir / config["data"]["train_path"]
    print(f"📂 加载 DPO 数据: {data_path}")

    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line.strip()))

    print(f"  共 {len(raw_data)} 个偏好对")

    # 转换为 DPO conversational 格式
    # DPOTrainer chat 模型要求 prompt/chosen/rejected 为 message 列表格式
    from datasets import Dataset

    # 加载工具 schema 构建系统提示
    with open(finetune_dir / "data" / "tools_schema.json", "r", encoding="utf-8") as f:
        tools_str = json.dumps(json.load(f), ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a GenAI Career Assistant. You have access to the following tools:\n\n"
        f"<tools>\n{tools_str}\n</tools>\n\n"
        "For each user request, decide whether to call a tool or respond directly.\n"
        "If calling a tool, output ONLY: <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n"
        "If the user's request is outside your capabilities, politely decline.\n"
        "If the user's request is missing required parameters, ask a follow-up question."
    )

    dpo_data = {
        "prompt": [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            for item in raw_data
        ],
        "chosen": [
            [{"role": "assistant", "content": item["chosen"]}]
            for item in raw_data
        ],
        "rejected": [
            [{"role": "assistant", "content": item["rejected"]}]
            for item in raw_data
        ],
    }
    dataset = Dataset.from_dict(dpo_data)

    # ── 3. DPO 训练 ──
    from trl import DPOTrainer, DPOConfig

    train_cfg = config["training"]
    output_dir = str(finetune_dir / train_cfg["output_dir"])

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        beta=train_cfg["beta"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        seed=config.get("seed", 42),
        report_to=config.get("report_to", "none"),
        optim="adamw_8bit",
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth/PEFT 模式下自动使用冻结的基座作 ref
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=dpo_config,
    )

    print(f"\n🚀 开始 DPO 训练...")
    print(f"  beta: {train_cfg['beta']}")
    print(f"  学习率: {train_cfg['learning_rate']}")
    trainer.train()

    # ── 4. 保存 ──
    print(f"\n💾 保存模型: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    info = {
        "base_model": model_name,
        "stage": "dpo",
        "beta": train_cfg["beta"],
        "train_pairs": len(raw_data),
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("✅ DPO 训练完成!")


if __name__ == "__main__":
    main()
