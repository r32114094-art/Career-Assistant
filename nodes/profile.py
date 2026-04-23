"""
nodes/profile.py - 用户画像提取节点

每轮对话结束后运行，从用户消息中提取画像信息并增量更新 state["user_profile"]。
- 短消息/纯指令直接跳过
- 检测到简历上传时走专用路径：提取简历正文段落 + 简历专用 prompt
- 普通对话走常规路径：只提取明确提到的字段
- 通过 merge_profile reducer 与历史画像合并
- 同步写回 user_store（跨会话持久化）
"""
import json
import re

from langchain_core.runnables import RunnableConfig

import user_store
from config import llm
from state import State, get_latest_user_text, merge_profile

_SKIP_TEXTS = {
    "继续", "好的", "好", "ok", "okay", "yes", "no", "嗯", "嗯嗯",
    "谢谢", "谢了", "下一步", "是的", "不", "continue", "go on", "next",
    "明白", "知道了", "收到", "好的好的",
}

# ── 普通对话提取 prompt ────────────────────────────────────────────────────────

_CHAT_EXTRACT_PROMPT = """\
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

# ── 简历专用提取 prompt ────────────────────────────────────────────────────────

_RESUME_EXTRACT_PROMPT = """\
你是一个简历解析专家。从下面的简历原文中提取用户的职业画像信息。

可提取的字段（尽可能完整提取，合理推断即可）：
- name: 姓名（从简历抬头/联系方式中提取）
- target_role: 目标职位（从求职意向/职位标题/最近工作推断，如 "算法工程师"、"NLP Engineer"）
- skill_level: 技能水平，根据工作年限和职位级别判断，只能是 beginner / intermediate / senior 之一
- years_of_experience: 总工作年限（整数，从工作经历时间段计算）
- background: 一句话背景摘要（教育 + 最近职位，不超过60字）
- skills: 技术技能列表（从技能栏、项目描述、工作经历中提取所有出现的技术词汇）
- interests: 技术兴趣/专长方向（从项目方向、研究方向、个人总结中推断）
- preferred_location: 求职地点（若简历中明确注明）
- preferred_work_type: 工作方式，只能是 remote / onsite / hybrid 之一（若明确注明）

规则：
1. skills 要尽可能完整，涵盖编程语言、框架、工具、平台
2. years_of_experience 从最早工作时间到现在计算，实习不算（若只有实习则算0）
3. skill_level 判断依据：0-1年beginner，2-4年intermediate，5年以上senior
4. 没有明确依据的字段返回 null，不要凭空捏造
5. 只输出 JSON，不要有任何其他文字

简历原文：
{resume_text}

已知画像（供参考，避免重复提取）：{existing}

JSON:"""


def _extract_resume_text(user_text: str) -> str | None:
    """从用户消息中提取 【简历内容】 段落，返回简历正文（最多4000字）。"""
    match = re.search(r'【简历内容】\s*([\s\S]+?)(?:【目标岗位 JD】|$)', user_text)
    if match:
        return match.group(1).strip()[:4000]
    return None


def _is_resume_message(user_text: str) -> bool:
    return '【简历内容】' in user_text


def _parse_json(raw: str) -> dict:
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


def _get_user_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("user_id") or ""


def update_profile(state: State, config: RunnableConfig) -> dict:
    """从最新用户消息中提取画像信息，增量写入 state["user_profile"] 并同步到 user_store。"""
    user_text = get_latest_user_text(state)
    user_id = _get_user_id(config)
    existing = state.get("user_profile") or {}
    existing_json = json.dumps(existing, ensure_ascii=False) if existing else "{}"

    if _is_resume_message(user_text):
        # ── 简历上传：专用高精度路径 ──────────────────────────────
        resume_text = _extract_resume_text(user_text)
        if not resume_text:
            return {}
        print(f"[Profile] 检测到简历上传，使用简历专用提取（{len(resume_text)} 字）")
        raw = llm.invoke(_RESUME_EXTRACT_PROMPT.format(
            resume_text=resume_text,
            existing=existing_json,
        )).content
    else:
        # ── 普通对话：轻量提取 ────────────────────────────────────
        text = user_text.strip()
        if len(text) < 15 or text.lower() in _SKIP_TEXTS:
            return {}
        raw = llm.invoke(_CHAT_EXTRACT_PROMPT.format(
            message=text[:800],
            existing=existing_json,
        )).content

    extracted = _parse_json(raw)
    if not extracted:
        return {}

    cleaned = {k: v for k, v in extracted.items() if v is not None and v != [] and v != ""}
    if not cleaned:
        return {}

    print(f"[Profile] 本轮提取到画像更新: {cleaned}")

    if user_id:
        persisted = user_store.get_profile(user_id)
        merged = merge_profile(persisted, cleaned)
        user_store.save_profile(user_id, merged)

    return {"user_profile": cleaned}
