"""
01b_split_dataset.py — 合并所有类别数据并拆分 train/val/test

用法:
  python scripts/01b_split_dataset.py --input data/sft/ --output data/sft/ --val-ratio 0.1 --test-ratio 0.1
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="合并并拆分 SFT 数据集")
    parser.add_argument("--input", type=str, default="data/sft/", help="输入目录（含 sft_*.jsonl 文件）")
    parser.add_argument("--output", type=str, default="data/sft/", help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    finetune_dir = Path(__file__).parent.parent
    input_dir = finetune_dir / args.input
    output_dir = finetune_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 sft_*.jsonl
    all_data = []
    category_counts = Counter()

    for f in sorted(input_dir.glob("sft_*.jsonl")):
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                item = json.loads(line.strip())
                all_data.append(item)
                cat = item.get("metadata", {}).get("category", "unknown")
                category_counts[cat] += 1

    print(f"📊 总数据量: {len(all_data)}")
    print(f"📊 类别分布:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")

    if len(all_data) == 0:
        print("❌ 未找到任何 sft_*.jsonl 文件，请先运行 01_generate_sft_data.py")
        return

    # 分层抽样：按 category 分组后各自拆分，确保 test 集覆盖所有类别
    from collections import defaultdict
    by_category = defaultdict(list)
    for item in all_data:
        cat = item.get("metadata", {}).get("category", "unknown")
        by_category[cat].append(item)

    train_data, val_data, test_data = [], [], []
    for cat, items in by_category.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * args.test_ratio))  # 每类至少 1 条进 test
        n_val = max(1, int(n * args.val_ratio))     # 每类至少 1 条进 val
        n_train = n - n_test - n_val

        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    # 各集内部再打乱
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # 保存
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"💾 {name}: {len(data)} 条 → {path}")

    # 验证类别分布
    print(f"\n📊 拆分后分布检查:")
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        dist = Counter(item.get("metadata", {}).get("category", "unknown") for item in data)
        print(f"  {name}: {dict(dist)}")


if __name__ == "__main__":
    main()
