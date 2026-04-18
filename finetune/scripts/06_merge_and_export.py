"""
06_merge_and_export.py — LoRA 合并 & 导出

将 LoRA adapter 合并回基座模型，导出为完整模型用于部署。

用法:
  python scripts/06_merge_and_export.py --checkpoint checkpoints/grpo_qwen25_3b --output models/fc_router_3b
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 到基座模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--output", type=str, required=True, help="合并后模型输出路径")
    parser.add_argument("--quantize", type=str, choices=["none", "q4_k_m", "q8_0"], default="none",
                        help="GGUF 量化等级 (可选)")
    args = parser.parse_args()

    finetune_dir = Path(__file__).parent.parent
    checkpoint_path = str(finetune_dir / args.checkpoint)
    output_path = str(finetune_dir / args.output)

    from unsloth import FastLanguageModel

    print(f"📦 加载 LoRA checkpoint: {checkpoint_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    # 合并 LoRA 权重并保存完整模型 (HuggingFace 16bit 格式)
    print(f"🔀 合并 LoRA adapter → {output_path}")
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    # 可选: GGUF 量化导出 (用于 llama.cpp / Ollama 部署)
    if args.quantize != "none":
        gguf_path = output_path + f"_{args.quantize}.gguf"
        print(f"📦 导出 GGUF 量化: {gguf_path}")
        model.save_pretrained_gguf(
            output_path,
            tokenizer,
            quantization_method=args.quantize,
        )

    print("✅ 导出完成!")


if __name__ == "__main__":
    main()
