"""
02_train_sft.py — SFT 训练脚本 (Unsloth)

用法 (在 AutoDL 上):
  pip install unsloth
  python scripts/02_train_sft.py --config configs/sft_config.yaml
"""

import json
import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_messages(path: str) -> list[dict]:
    """加载 JSONL 并提取 messages 字段。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if "messages" in item:
                data.append({"messages": item["messages"]})
    return data


def main():
    parser = argparse.ArgumentParser(description="SFT 训练 (Unsloth)")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()

    # 切换到 finetune 目录
    finetune_dir = Path(__file__).parent.parent
    config = load_config(finetune_dir / args.config)

    # ── 1. 加载模型 (Unsloth 加速) ──
    from unsloth import FastLanguageModel

    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]

    print(f"📦 加载模型: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=(config["lora"]["quantization"] == "4bit"),
        dtype=None,  # auto
    )

    # ── 2. 配置 LoRA ──
    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 42),
    )

    # ── 3. 加载数据 ──
    train_path = finetune_dir / config["data"]["train_path"]
    val_path = finetune_dir / config["data"]["val_path"]

    print(f"📂 加载训练数据: {train_path}")
    train_data = load_jsonl_messages(str(train_path))
    print(f"  训练集: {len(train_data)} 条")

    val_data = None
    if val_path.exists():
        val_data = load_jsonl_messages(str(val_path))
        print(f"  验证集: {len(val_data)} 条")

    # 转换为 Hugging Face Dataset
    from datasets import Dataset

    def format_for_training(examples):
        """将 messages 列表转换为 tokenizer 可处理的格式。"""
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_for_training, batched=True, remove_columns=["messages"])

    eval_dataset = None
    if val_data:
        eval_dataset = Dataset.from_list(val_data)
        eval_dataset = eval_dataset.map(format_for_training, batched=True, remove_columns=["messages"])

    # ── 4. 训练 ──
    # TRL>=1.0: dataset_text_field / max_seq_length / packing 必须通过 SFTConfig 传递
    from trl import SFTTrainer, SFTConfig

    train_cfg = config["training"]
    output_dir = str(finetune_dir / train_cfg["output_dir"])

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=config.get("seed", 42),
        report_to=config.get("report_to", "none"),
        optim="adamw_8bit",
        # SFT 特有参数（TRL>=1.0 必须在 Config 中设置）
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    print(f"\n🚀 开始 SFT 训练...")
    print(f"  输出目录: {output_dir}")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  有效 batch size: {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  学习率: {train_cfg['learning_rate']}")

    trainer.train()

    # ── 5. 保存 ──
    print(f"\n💾 保存模型到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练信息
    info = {
        "base_model": model_name,
        "lora_rank": lora_config["rank"],
        "train_samples": len(train_data),
        "epochs": train_cfg["epochs"],
        "stage": "sft",
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("✅ SFT 训练完成!")


if __name__ == "__main__":
    main()
