"""
eval/eval_routing.py - 路由准确率评测脚本

功能：
  1. 加载 routing_dataset.json 中的 120 条标注样本
  2. 对每条样本调用 categorize() 节点获取主分类
  3. 对学习类(1)和面试类(3)样本，额外调用子分类节点
  4. 与标注答案对比，输出混淆矩阵、准确率、F1 值、延迟统计
  5. 自动集成 LangSmith Tracing（如环境变量已配置）

运行方式：
  conda activate genai-project
  cd "GenAI Career Assistant"
  python eval/eval_routing.py

输出文件：
  eval/results/routing_report.txt    — 文字报告
  eval/results/confusion_matrix.csv  — 混淆矩阵
  eval/results/detail_log.csv        — 每条样本的详细判定结果
"""
import json
import os
import re
import sys
import time
import csv
from collections import defaultdict
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# 设置 LangSmith 追踪（如果环境变量已配置）
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "routing-eval")
    print("[LangSmith] Tracing enabled, project=routing-eval")
else:
    print("[LangSmith] No API key found, tracing disabled")

from langchain_core.messages import HumanMessage, AIMessage
from nodes.categorize import categorize, handle_learning_resource, handle_interview_preparation
from router import route_query, route_interview, route_learning


# ── 工具函数 ────────────────────────────────────────────

def build_state_from_sample(sample: dict) -> dict:
    """将评测样本转换为 LangGraph State 格式"""
    messages = []
    # 注入历史上下文
    for ctx in sample.get("context", []):
        if ctx["role"] == "user":
            messages.append(HumanMessage(content=ctx["content"]))
        else:
            messages.append(AIMessage(content=ctx["content"]))
    # 最后追加当前 query
    messages.append(HumanMessage(content=sample["query"]))
    return {"messages": messages, "category": "", "response": "", "pending_job_results": ""}


def extract_main_category(category_str: str) -> str:
    """从 categorize() 返回的字符串中提取 1-5 数字"""
    match = re.search(r"[1-5]", category_str)
    return match.group() if match else "?"


def extract_sub_category(category_str: str) -> str:
    """从子分类节点返回中提取 tutorial/question/mock"""
    lower = category_str.lower().strip()
    if "tutorial" in lower:
        return "tutorial"
    elif "mock" in lower:
        return "mock"
    else:
        return "question"


# ── 主评测逻辑 ──────────────────────────────────────────

def run_evaluation():
    dataset_path = PROJECT_ROOT / "eval" / "routing_dataset.json"
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"  Routing Accuracy Evaluation")
    print(f"  Dataset: {total} samples")
    print(f"{'='*60}\n")

    # 结果收集
    main_correct = 0
    sub_correct = 0
    sub_total = 0
    detail_log = []
    main_confusion = defaultdict(lambda: defaultdict(int))
    latencies = []
    errors = []

    for i, sample in enumerate(dataset):
        sid = sample["id"]
        query = sample["query"]
        expected_main = sample["expected_main"]
        expected_sub = sample.get("expected_sub")
        has_context = len(sample.get("context", [])) > 0

        # ── Step 1: 主分类 ──
        state = build_state_from_sample(sample)

        t0 = time.time()
        try:
            result = categorize(state)
        except Exception as e:
            errors.append({"id": sid, "error": str(e)})
            print(f"  [{sid:3d}] ERROR: {e}")
            detail_log.append({
                "id": sid, "query": query[:40], "context": has_context,
                "expected_main": expected_main, "predicted_main": "ERR",
                "main_correct": False,
                "expected_sub": expected_sub, "predicted_sub": "",
                "sub_correct": None, "latency_ms": 0
            })
            continue
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        predicted_main = extract_main_category(result.get("category", ""))
        is_main_correct = (predicted_main == expected_main)
        if is_main_correct:
            main_correct += 1

        main_confusion[expected_main][predicted_main] += 1

        # ── Step 2: 子分类（仅对学习类和面试类） ──
        predicted_sub = ""
        is_sub_correct = None

        if expected_sub and is_main_correct:
            sub_total += 1
            state["category"] = result["category"]

            try:
                if expected_main == "1":  # 学习
                    sub_result = handle_learning_resource(state)
                elif expected_main == "3":  # 面试
                    sub_result = handle_interview_preparation(state)
                else:
                    sub_result = None

                if sub_result:
                    predicted_sub = extract_sub_category(sub_result.get("category", ""))
                    is_sub_correct = (predicted_sub == expected_sub)
                    if is_sub_correct:
                        sub_correct += 1
            except Exception as e:
                errors.append({"id": sid, "error": f"sub: {e}"})

        # ── 打印进度 ──
        status = "OK" if is_main_correct else "MISS"
        sub_status = ""
        if is_sub_correct is not None:
            sub_status = f" | sub={'OK' if is_sub_correct else 'MISS'}"
        ctx_tag = " [CTX]" if has_context else ""
        print(f"  [{sid:3d}/{total}] {status}{sub_status}{ctx_tag}  "
              f"expect={expected_main} got={predicted_main}  "
              f"{latency_ms:.0f}ms  {query[:35]}...")

        detail_log.append({
            "id": sid, "query": query[:60], "context": has_context,
            "expected_main": expected_main, "predicted_main": predicted_main,
            "main_correct": is_main_correct,
            "expected_sub": expected_sub or "", "predicted_sub": predicted_sub,
            "sub_correct": is_sub_correct, "latency_ms": round(latency_ms, 1)
        })

    # ── 生成报告 ────────────────────────────────────────

    main_accuracy = main_correct / total * 100 if total > 0 else 0
    sub_accuracy = sub_correct / sub_total * 100 if sub_total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    # ── 计算各类别 Precision / Recall / F1 ──
    categories = ["1", "2", "3", "4", "5"]
    category_names = {
        "1": "Learning", "2": "Resume", "3": "Interview",
        "4": "JobSearch", "5": "OutOfScope"
    }
    per_class_metrics = {}
    for cat in categories:
        tp = main_confusion[cat][cat]
        fp = sum(main_confusion[other][cat] for other in categories if other != cat)
        fn = sum(main_confusion[cat][other] for other in categories if other != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_metrics[cat] = {
            "name": category_names[cat],
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn
        }

    # ── 打印报告 ──
    report_lines = []
    def rprint(line=""):
        report_lines.append(line)
        print(line)

    rprint(f"\n{'='*60}")
    rprint(f"  EVALUATION RESULTS")
    rprint(f"{'='*60}")
    rprint(f"  Total samples:     {total}")
    rprint(f"  Main accuracy:     {main_correct}/{total} = {main_accuracy:.1f}%")
    rprint(f"  Sub accuracy:      {sub_correct}/{sub_total} = {sub_accuracy:.1f}%")
    rprint(f"  Avg latency:       {avg_latency:.0f} ms")
    rprint(f"  P95 latency:       {p95_latency:.0f} ms")
    rprint(f"  Errors:            {len(errors)}")
    rprint()
    rprint(f"  {'Category':<14} {'Prec':>6} {'Recall':>7} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    rprint(f"  {'-'*50}")
    for cat in categories:
        m = per_class_metrics[cat]
        rprint(f"  {m['name']:<14} {m['precision']:>6.1%} {m['recall']:>7.1%} {m['f1']:>6.1%} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    rprint(f"{'='*60}")

    # ── Context vs Non-context 分析 ──
    ctx_samples = [d for d in detail_log if d["context"]]
    no_ctx_samples = [d for d in detail_log if not d["context"]]
    ctx_acc = sum(1 for d in ctx_samples if d["main_correct"]) / len(ctx_samples) * 100 if ctx_samples else 0
    no_ctx_acc = sum(1 for d in no_ctx_samples if d["main_correct"]) / len(no_ctx_samples) * 100 if no_ctx_samples else 0
    rprint(f"\n  Context-dependent accuracy:  {ctx_acc:.1f}% ({len(ctx_samples)} samples)")
    rprint(f"  No-context accuracy:         {no_ctx_acc:.1f}% ({len(no_ctx_samples)} samples)")

    # ── 错误案例 ──
    miss_cases = [d for d in detail_log if not d["main_correct"]]
    if miss_cases:
        rprint(f"\n  === Misclassified Cases ({len(miss_cases)} total) ===")
        for mc in miss_cases:
            rprint(f"  ID={mc['id']}  expect={mc['expected_main']} got={mc['predicted_main']}  "
                   f"ctx={mc['context']}  query={mc['query']}")

    # ── 保存文件 ──

    # 1. 文字报告
    report_path = results_dir / "routing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  [SAVED] {report_path}")

    # 2. 混淆矩阵 CSV
    cm_path = results_dir / "confusion_matrix.csv"
    with open(cm_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + categories)
        for actual in categories:
            row = [actual] + [main_confusion[actual][pred] for pred in categories]
            writer.writerow(row)
    print(f"  [SAVED] {cm_path}")

    # 3. 详细日志 CSV
    log_path = results_dir / "detail_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "query", "context", "expected_main", "predicted_main",
            "main_correct", "expected_sub", "predicted_sub", "sub_correct", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(detail_log)
    print(f"  [SAVED] {log_path}")

    print(f"\n  Done! Main accuracy = {main_accuracy:.1f}%\n")
    return main_accuracy


if __name__ == "__main__":
    run_evaluation()
