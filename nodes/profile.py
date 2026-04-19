"""
nodes/profile.py - 用户画像提取节点

每轮对话结束后运行，从用户消息中提取画像信息并增量更新 state["user_profile"]。
- 短消息/纯指令直接跳过，不消耗 LLM
- 只提取本轮消息中明确提到的字段，其余返回 null
- 通过 merge_profile reducer 与历史画像合并，不覆盖已有信息
"""
import json
import re

from config import llm
from state import State, get_latest_user_text

_SKIP_TEXTS = {
    "继续", "好的", "好", "ok", "okay", "yes", "no", "嗯", "嗯嗯",
    "谢谢", "谢了", "下一步", "是的", "不", "continue", "go on", "next",
    "明白", "知道了", "收到", "好的好的",
}

_EXTRACT_PROMPT = """\
你是一个信息提取助手。从下面的用户消息中提取职业相关信息，只提取消息中明确提到的内容。

可提取的字段：
- name: 用户姓名（字符串）
- target_role: 目标职位，如 "NLP Engineer"、"AI产品经理"（字符串）
- skill_level: 技能水平，只能是 beginner / intermediate / senior 之一
- years_of_experience: 工作年限（整数）
- background: 教育或工作背景的简短描述（字符串，不超过50字）
- skills: 技术技能列表，如 ["Python", "PyTorch", "RAG"]
- interests: 技术兴趣方向列表，如 ["LLM fine-tuning", "Agent"]
- preferred_location: 求职地点偏好（字符串）
- preferred_work_type: 工作方式，只能是 remote / onsite / hybrid 之一

规则：
1. 只提取消息中明确出现的信息，不要推断或猜测
2. 未提及的字段一律返回 null
3. 只输出 JSON，不要有任何其他文字

用户消息：{message}

已知画像（供参考，避免重复提取）：{existing}

JSON:"""


def update_profile(state: State) -> dict:
    """从最新用户消息中提取画像信息，增量写入 state["user_profile"]。"""
    user_text = get_latest_user_text(state)

    # 规则过滤：短消息或纯指令直接跳过
    if len(user_text.strip()) < 15 or user_text.strip().lower() in _SKIP_TEXTS:
        return {}

    existing = state.get("user_profile") or {}

    raw = llm.invoke(_EXTRACT_PROMPT.format(
        message=user_text[:500],
        existing=json.dumps(existing, ensure_ascii=False) if existing else "{}",
    )).content

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return {}

    try:
        extracted = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}

    # 过滤掉 null 值，只保留有实际内容的字段
    cleaned = {k: v for k, v in extracted.items() if v is not None and v != [] and v != ""}

    if not cleaned:
        return {}

    print(f"[Profile] 本轮提取到画像更新: {cleaned}")
    return {"user_profile": cleaned}
