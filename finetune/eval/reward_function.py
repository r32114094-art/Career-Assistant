"""
GRPO 奖励函数 — 多维度打分机制

为 Function Call 任务设计的可验证奖励函数，用于 GRPO 训练阶段。
同时也可作为各阶段的评估工具。

打分维度:
  - 格式合规 (0.2): 输出是否为可解析的 <tool_call> JSON
  - 工具选择 (0.3): 是否选对了工具
  - 必填参数 (0.3): required 参数是否正确提取
  - 可选参数 (0.1): optional 参数的准确性
  - 无幻觉   (0.1): 是否存在 schema 之外的幻觉参数
"""

import json
import re
from typing import Optional


# ── 解析工具 ──────────────────────────────────────────

def parse_tool_call(text: str) -> Optional[dict]:
    """从模型输出中提取 <tool_call> JSON。

    支持的格式:
      1. <tool_call>{"name":"...", "arguments":{...}}</tool_call>
      2. 裸 JSON（兜底）

    Returns:
        解析成功: {"name": str, "arguments": dict}
        解析失败: None
    """
    # 尝试匹配 <tool_call> 标签
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # 兜底：用括号计数法从文本中提取完整 JSON 对象
    for i, ch in enumerate(text):
        if ch == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                if depth == 0:
                    candidate = text[i:j+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break  # 这个起始位置的 JSON 无效，继续找下一个 '{'

    return None


def is_rejection(text: str) -> bool:
    """判断模型是否做了拒绝回复（没有调用任何工具）。"""
    return parse_tool_call(text) is None


# ── 核心奖励函数 ──────────────────────────────────────

def tool_call_reward(
    model_output: str,
    ground_truth: dict,
    tool_names: list[str] = None,
) -> dict:
    """多维度 Function Call 奖励函数。

    Args:
        model_output: 模型生成的原始文本
        ground_truth: 标准答案字典，格式:
            {
                "action": "call" | "reject" | "ask_followup",
                "name": "search_jobs",               # action=call 时必填
                "required_params": {"role": "...", "location": "..."},
                "optional_params": {"job_type": "remote"},
            }
        tool_names: 合法的工具名称列表

    Returns:
        dict: {
            "total": float (0.0 ~ 1.0),
            "breakdown": {维度: 分数},
            "errors": [错误描述列表]
        }
    """
    breakdown = {
        "format": 0.0,
        "tool_selection": 0.0,
        "required_params": 0.0,
        "optional_params": 0.0,
        "no_hallucination": 0.0,
    }
    errors = []
    expected_action = ground_truth.get("action", "call")

    # ── 特殊情况: 期望拒绝或追问 ──
    if expected_action in ("reject", "ask_followup"):
        if is_rejection(model_output):
            # 正确拒绝/追问
            breakdown["format"] = 0.2
            breakdown["tool_selection"] = 0.3
            breakdown["required_params"] = 0.3
            breakdown["no_hallucination"] = 0.1

            # 追问场景：检查是否包含问号或追问语气
            if expected_action == "ask_followup":
                if "?" in model_output or "？" in model_output:
                    breakdown["optional_params"] = 0.1
                else:
                    errors.append("追问但缺少问句")
            else:
                breakdown["optional_params"] = 0.1

            return {
                "total": sum(breakdown.values()),
                "breakdown": breakdown,
                "errors": errors,
            }
        else:
            # 不该调用但调用了
            errors.append(f"应该{expected_action}但错误调用了工具")
            return {
                "total": 0.0,
                "breakdown": breakdown,
                "errors": errors,
            }

    # ── 正常情况: 期望调用工具 ──
    parsed = parse_tool_call(model_output)

    # 维度 1: 格式合规 (0.2)
    if parsed is None:
        errors.append("输出不包含可解析的 tool_call JSON")
        return {"total": 0.0, "breakdown": breakdown, "errors": errors}
    breakdown["format"] = 0.2

    # 维度 2: 工具选择 (0.3)
    expected_name = ground_truth.get("name", "")
    if parsed["name"] == expected_name:
        breakdown["tool_selection"] = 0.3
    else:
        errors.append(f"工具选择错误: 预期 {expected_name}, 实际 {parsed['name']}")
        # 工具选错，参数分意义不大，但仍然统计
        return {
            "total": sum(breakdown.values()),
            "breakdown": breakdown,
            "errors": errors,
        }

    # 维度 3: 必填参数 (0.3)
    required_params = ground_truth.get("required_params", {})
    if required_params:
        correct = 0
        for k, v in required_params.items():
            actual = parsed["arguments"].get(k)
            if actual is not None and _param_match(actual, v):
                correct += 1
            else:
                errors.append(f"必填参数 {k}: 预期 {v!r}, 实际 {actual!r}")
        breakdown["required_params"] = 0.3 * (correct / len(required_params))
    else:
        breakdown["required_params"] = 0.3

    # 维度 4: 可选参数 (0.1)
    optional_params = ground_truth.get("optional_params", {})
    if optional_params:
        correct = 0
        for k, v in optional_params.items():
            actual = parsed["arguments"].get(k)
            if actual is not None and _param_match(actual, v):
                correct += 1
        breakdown["optional_params"] = 0.1 * (correct / len(optional_params))
    else:
        breakdown["optional_params"] = 0.1

    # 维度 5: 无幻觉参数 (0.1)
    all_valid_keys = set(required_params) | set(optional_params)
    hallucinated = set(parsed["arguments"]) - all_valid_keys
    if not hallucinated:
        breakdown["no_hallucination"] = 0.1
    else:
        errors.append(f"幻觉参数: {hallucinated}")

    return {
        "total": sum(breakdown.values()),
        "breakdown": breakdown,
        "errors": errors,
    }


def _param_match(actual, expected) -> bool:
    """参数值匹配——支持语义宽松比较。

    规则:
      - 字符串: 忽略大小写和首尾空白
      - 列表: 转为集合后比较（忽略顺序）
      - 数值: 直接相等
    """
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    if isinstance(expected, list) and isinstance(actual, list):
        return set(str(x).lower() for x in actual) == set(str(x).lower() for x in expected)
    return actual == expected


# ── GRPO 兼容接口 ──────────────────────────────────────

def grpo_reward_fn(completions: list[str], ground_truths: list[dict], **kwargs) -> list[float]:
    """TRL GRPOTrainer 兼容的批量奖励函数。

    Args:
        completions: 一组模型生成的文本
        ground_truths: 对应的标准答案

    Returns:
        list[float]: 奖励分数列表
    """
    return [
        tool_call_reward(c, gt)["total"]
        for c, gt in zip(completions, ground_truths)
    ]


# ── 评估统计 ──────────────────────────────────────────

def evaluate_batch(
    predictions: list[str],
    ground_truths: list[dict],
) -> dict:
    """批量评估并返回统计指标。

    Returns:
        dict: {
            "tool_selection_accuracy": float,
            "param_extraction_f1": float,
            "format_compliance_rate": float,
            "rejection_accuracy": float,
            "false_positive_rate": float,
            "avg_reward": float,
            "per_sample": [每条样本的详细结果],
        }
    """
    results = []
    for pred, gt in zip(predictions, ground_truths):
        r = tool_call_reward(pred, gt)
        results.append(r)

    total = len(results)
    if total == 0:
        return {}

    # 工具选择准确率
    tool_correct = sum(1 for r in results if r["breakdown"]["tool_selection"] == 0.3)
    tool_selection_accuracy = tool_correct / total

    # 格式合规率
    format_ok = sum(1 for r in results if r["breakdown"]["format"] == 0.2)
    format_compliance_rate = format_ok / total

    # 参数提取 F1 (简化: 用 required_params 的平均得分率)
    param_scores = [r["breakdown"]["required_params"] / 0.3 for r in results if r["breakdown"]["tool_selection"] == 0.3]
    param_extraction_f1 = sum(param_scores) / len(param_scores) if param_scores else 0.0

    # 拒绝准确率
    reject_samples = [
        (r, gt) for r, gt in zip(results, ground_truths)
        if gt.get("action") in ("reject", "ask_followup")
    ]
    if reject_samples:
        reject_correct = sum(1 for r, _ in reject_samples if r["total"] >= 0.8)
        rejection_accuracy = reject_correct / len(reject_samples)
    else:
        rejection_accuracy = None

    # 误报率
    non_call_gts = [(r, gt) for r, gt in zip(results, ground_truths) if gt.get("action") != "call"]
    if non_call_gts:
        false_positives = sum(1 for r, _ in non_call_gts if r["breakdown"]["format"] == 0.2)
        false_positive_rate = false_positives / len(non_call_gts)
    else:
        false_positive_rate = None

    avg_reward = sum(r["total"] for r in results) / total

    return {
        "tool_selection_accuracy": round(tool_selection_accuracy, 4),
        "param_extraction_f1": round(param_extraction_f1, 4),
        "format_compliance_rate": round(format_compliance_rate, 4),
        "rejection_accuracy": round(rejection_accuracy, 4) if rejection_accuracy is not None else "N/A",
        "false_positive_rate": round(false_positive_rate, 4) if false_positive_rate is not None else "N/A",
        "avg_reward": round(avg_reward, 4),
        "total_samples": total,
        "per_sample": results,
    }


# ── 测试入口 ──────────────────────────────────────────

if __name__ == "__main__":
    # 快速自测
    test_cases = [
        # Case 1: 完美调用
        (
            '<tool_call>{"name":"search_jobs","arguments":{"role":"AI工程师","location":"上海"}}</tool_call>',
            {"action": "call", "name": "search_jobs", "required_params": {"role": "AI工程师", "location": "上海"}},
        ),
        # Case 2: 工具选错
        (
            '<tool_call>{"name":"ask_ai_question","arguments":{"question":"面试准备"}}</tool_call>',
            {"action": "call", "name": "get_interview_questions", "required_params": {"topic": "面试准备"}},
        ),
        # Case 3: 正确拒绝
        (
            "抱歉，我是 GenAI 职业助手，不支持天气查询。",
            {"action": "reject"},
        ),
        # Case 4: 应该拒绝但调用了
        (
            '<tool_call>{"name":"ask_ai_question","arguments":{"question":"今天天气怎么样"}}</tool_call>',
            {"action": "reject"},
        ),
        # Case 5: 参数幻觉
        (
            '<tool_call>{"name":"search_jobs","arguments":{"role":"AI工程师","location":"上海","salary":"30k"}}</tool_call>',
            {"action": "call", "name": "search_jobs", "required_params": {"role": "AI工程师", "location": "上海"}},
        ),
    ]

    print("=" * 60)
    print("奖励函数自测")
    print("=" * 60)
    for i, (output, gt) in enumerate(test_cases):
        result = tool_call_reward(output, gt)
        print(f"\nCase {i+1}: total={result['total']:.2f}")
        print(f"  breakdown: {result['breakdown']}")
        if result["errors"]:
            print(f"  errors: {result['errors']}")
