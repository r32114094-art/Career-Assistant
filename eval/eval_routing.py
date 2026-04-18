"""
eval/eval_routing.py - 路由准确率评测脚本（结构化路由版）

功能：
  1. 加载 routing_dataset.json 中的 120 条标注样本
  2. 对每条样本调用 categorize() 节点，读取结构化 routing_decision 字段
  3. 对学习类和面试类样本，额外调用子分类节点
  4. 与标注答案对比，输出混淆矩阵、准确率、F1 值、延迟统计
  5. 新增指标：解析成功率、低置信度触发率、澄清分支触发率

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
import sys
import time
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "routing-eval")
    print("[LangSmith] Tracing enabled, project=routing-eval")
else:
    print("[LangSmith] No API key found, tracing disabled")

from langchain_core.messages import HumanMessage, AIMessage
from nodes.categorize import categorize, handle_learning_resource, handle_interview_preparation
from router import CONFIDENCE_THRESHOLD

# ── 枚举映射（数据集用数字，新系统用字符串） ─────────────────
MAIN_NUM_TO_INTENT = {
    "1": "learning",
    "2": "resume",
    "3": "interview",
    "4": "job_search",
    "5": "out_of_scope",
}
MAIN_INTENT_TO_NUM = {v: k for k, v in MAIN_NUM_TO_INTENT.items()}
VALID_MAIN = set(MAIN_NUM_TO_INTENT.values())


# ── 工具函数 ────────────────────────────────────────────────

def build_state_from_sample(sample: dict) -> dict:
    """将评测样本转换为 LangGraph State 格式"""
    messages = []
    for ctx in sample.get("context", []):
        if ctx["role"] == "user":
            messages.append(HumanMessage(content=ctx["content"]))
        else:
            messages.append(AIMessage(content=ctx["content"]))
    messages.append(HumanMessage(content=sample["query"]))
    return {
        "messages": messages,
        "category": "",
        "routing_decision": None,
        "clarify_count": 0,
        "response": "",
        "pending_job_results": "",
    }


# ── 主评测逻辑 ──────────────────────────────────────────────

def run_evaluation():
    dataset_path = PROJECT_ROOT / "eval" / "routing_dataset.json"
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"  Routing Accuracy Evaluation (Structured Mode)")
    print(f"  Dataset: {total} samples  |  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}\n")

    main_correct = 0
    sub_correct = 0
    sub_total = 0
    parse_success_count = 0
    low_conf_count = 0
    clarify_count_total = 0
    detail_log = []
    main_confusion = defaultdict(lambda: defaultdict(int))
    latencies = []
    errors = []
    confidence_by_intent = defaultdict(list)

    for i, sample in enumerate(dataset):
        sid = sample["id"]
        query = sample["query"]
        expected_main_num = sample["expected_main"]
        expected_sub = sample.get("expected_sub")
        has_context = len(sample.get("context", [])) > 0

        state = build_state_from_sample(sample)

        # ── Step 1: 主分类 ──
        t0 = time.time()
        try:
            result = categorize(state)
        except Exception as e:
            errors.append({"id": sid, "error": str(e)})
            print(f"  [{sid:3d}] ERROR: {e}")
            detail_log.append({
                "id": sid, "query": query[:40], "context": has_context,
                "expected_main": expected_main_num, "predicted_main": "ERR",
                "main_correct": False, "parse_success": False,
                "confidence": 0.0, "needs_clarification": False,
                "expected_sub": expected_sub, "predicted_sub": "",
                "sub_correct": None, "latency_ms": 0, "reason": "exception",
            })
            continue
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # ── 读取结构化字段（不再用正则） ──
        rd = result.get("routing_decision") or {}
        predicted_intent = rd.get("main_intent", "")
        predicted_main_num = MAIN_INTENT_TO_NUM.get(predicted_intent, "?")
        confidence = float(rd.get("confidence", 0.0))
        needs_clarification = bool(rd.get("needs_clarification", False))
        reason = rd.get("reason", "")

        parse_success = predicted_intent in VALID_MAIN
        if parse_success:
            parse_success_count += 1
        if confidence < CONFIDENCE_THRESHOLD:
            low_conf_count += 1
        if needs_clarification:
            clarify_count_total += 1
        confidence_by_intent[predicted_intent].append(confidence)

        is_main_correct = (predicted_main_num == expected_main_num)
        if is_main_correct:
            main_correct += 1
        main_confusion[expected_main_num][predicted_main_num] += 1

        # ── Step 2: 子分类（仅对学习类和面试类） ──
        predicted_sub = ""
        is_sub_correct = None

        if expected_sub and is_main_correct:
            sub_total += 1
            state["routing_decision"] = result.get("routing_decision")
            try:
                if expected_main_num == "1":
                    sub_result = handle_learning_resource(state)
                elif expected_main_num == "3":
                    sub_result = handle_interview_preparation(state)
                else:
                    sub_result = None

                if sub_result:
                    rd_sub = sub_result.get("routing_decision") or {}
                    predicted_sub = rd_sub.get("sub_intent", "question")
                    is_sub_correct = (predicted_sub == expected_sub)
                    if is_sub_correct:
                        sub_correct += 1
            except Exception as e:
                errors.append({"id": sid, "error": f"sub: {e}"})

        status = "OK" if is_main_correct else "MISS"
        sub_status = f" | sub={'OK' if is_sub_correct else 'MISS'}" if is_sub_correct is not None else ""
        ctx_tag = " [CTX]" if has_context else ""
        parse_tag = "" if parse_success else " [PARSE_FAIL]"
        print(f"  [{sid:3d}/{total}] {status}{sub_status}{ctx_tag}{parse_tag}  "
              f"expect={expected_main_num} got={predicted_main_num}  "
              f"conf={confidence:.2f}  {latency_ms:.0f}ms  {query[:30]}...")

        detail_log.append({
            "id": sid, "query": query[:60], "context": has_context,
            "expected_main": expected_main_num, "predicted_main": predicted_main_num,
            "main_correct": is_main_correct, "parse_success": parse_success,
            "confidence": round(confidence, 3), "needs_clarification": needs_clarification,
            "expected_sub": expected_sub or "", "predicted_sub": predicted_sub,
            "sub_correct": is_sub_correct, "latency_ms": round(latency_ms, 1),
            "reason": reason,
        })

    # ── 生成报告 ────────────────────────────────────────────
    main_accuracy = main_correct / total * 100 if total else 0
    sub_accuracy = sub_correct / sub_total * 100 if sub_total else 0
    parse_success_rate = parse_success_count / total * 100 if total else 0
    low_conf_rate = low_conf_count / total * 100 if total else 0
    clarify_rate = clarify_count_total / total * 100 if total else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

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
            "tp": tp, "fp": fp, "fn": fn,
        }

    report_lines = []
    def rprint(line=""):
        report_lines.append(line)
        print(line)

    rprint(f"\n{'='*60}")
    rprint(f"  EVALUATION RESULTS (Structured Routing)")
    rprint(f"{'='*60}")
    rprint(f"  Total samples:          {total}")
    rprint(f"  Main accuracy:          {main_correct}/{total} = {main_accuracy:.1f}%")
    rprint(f"  Sub accuracy:           {sub_correct}/{sub_total} = {sub_accuracy:.1f}%")
    rprint(f"  Avg latency:            {avg_latency:.0f} ms")
    rprint(f"  P95 latency:            {p95_latency:.0f} ms")
    rprint(f"  Errors:                 {len(errors)}")
    rprint()
    rprint(f"  --- New Metrics ---")
    rprint(f"  Parse success rate:     {parse_success_count}/{total} = {parse_success_rate:.1f}%")
    rprint(f"  Low confidence rate:    {low_conf_count}/{total} = {low_conf_rate:.1f}%  (threshold={CONFIDENCE_THRESHOLD})")
    rprint(f"  Clarify triggered rate: {clarify_count_total}/{total} = {clarify_rate:.1f}%")
    rprint()
    rprint(f"  {'Category':<14} {'Prec':>6} {'Recall':>7} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'AvgConf':>8}")
    rprint(f"  {'-'*58}")
    for cat in categories:
        m = per_class_metrics[cat]
        intent = MAIN_NUM_TO_INTENT[cat]
        confs = confidence_by_intent.get(intent, [])
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        rprint(f"  {m['name']:<14} {m['precision']:>6.1%} {m['recall']:>7.1%} {m['f1']:>6.1%} "
               f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {avg_conf:>8.2f}")
    rprint(f"{'='*60}")

    ctx_samples = [d for d in detail_log if d["context"]]
    no_ctx_samples = [d for d in detail_log if not d["context"]]
    ctx_acc = sum(1 for d in ctx_samples if d["main_correct"]) / len(ctx_samples) * 100 if ctx_samples else 0
    no_ctx_acc = sum(1 for d in no_ctx_samples if d["main_correct"]) / len(no_ctx_samples) * 100 if no_ctx_samples else 0
    rprint(f"\n  Context-dependent accuracy:  {ctx_acc:.1f}% ({len(ctx_samples)} samples)")
    rprint(f"  No-context accuracy:         {no_ctx_acc:.1f}% ({len(no_ctx_samples)} samples)")

    # ── 误差分桶 ──
    miss_cases = [d for d in detail_log if not d["main_correct"]]
    parse_fail_cases = [d for d in detail_log if not d["parse_success"]]
    if miss_cases:
        rprint(f"\n  === Misclassified Cases ({len(miss_cases)} total) ===")
        for mc in miss_cases:
            rprint(f"  ID={mc['id']}  expect={mc['expected_main']} got={mc['predicted_main']}  "
                   f"conf={mc['confidence']:.2f}  ctx={mc['context']}  query={mc['query']}")
    if parse_fail_cases:
        rprint(f"\n  === Parse Failures ({len(parse_fail_cases)} total) ===")
        for pf in parse_fail_cases:
            rprint(f"  ID={pf['id']}  reason={pf['reason']}  query={pf['query']}")

    # ── 保存文件 ──
    report_path = results_dir / "routing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  [SAVED] {report_path}")

    cm_path = results_dir / "confusion_matrix.csv"
    with open(cm_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + categories)
        for actual in categories:
            row = [actual] + [main_confusion[actual][pred] for pred in categories]
            writer.writerow(row)
    print(f"  [SAVED] {cm_path}")

    log_path = results_dir / "detail_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "query", "context", "expected_main", "predicted_main",
            "main_correct", "parse_success", "confidence", "needs_clarification",
            "expected_sub", "predicted_sub", "sub_correct", "latency_ms", "reason",
        ])
        writer.writeheader()
        writer.writerows(detail_log)
    print(f"  [SAVED] {log_path}")

    print(f"\n  Done! Main={main_accuracy:.1f}%  ParseOK={parse_success_rate:.1f}%  LowConf={low_conf_rate:.1f}%\n")
    return main_accuracy


if __name__ == "__main__":
    run_evaluation()
