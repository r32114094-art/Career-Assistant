"""
nodes/categorize.py - 分类节点

包含节点函数：
- categorize:                   将用户查询归类为结构化 RoutingDecision（主分类）
- handle_learning_resource:     将学习类查询细分为 tutorial / question
- handle_interview_preparation: 将面试类查询细分为 mock / question
- out_of_scope:                 处理不支持的查询
"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

import user_store
from config import llm
from state import State, RoutingDecision, get_latest_user_text


# ── 合法枚举值 ────────────────────────────────────────────────────────────────
VALID_MAIN = {"learning", "resume", "interview", "job_search", "out_of_scope"}
VALID_SUB = {"question", "tutorial", "mock", "null"}

# ── 主分类 Prompt ─────────────────────────────────────────────────────────────
_CATEGORIZE_PROMPT = ChatPromptTemplate.from_template(
    "You are a routing classifier. Your ONLY output must be a valid JSON object.\n\n"
    "Categories:\n"
    '- "learning": AI/ML questions, tutorials, blogs, guides, educational content about AI\n'
    '- "resume": reviewing, improving, or analyzing existing resumes; resume feedback, CV optimization\n'
    '- "interview": interview questions, tips, mock interviews, career advice related to interviews\n'
    '- "job_search": searching for jobs, finding openings, asking about hiring companies\n'
    '- "out_of_scope": completely unrelated to AI careers (cooking, weather, games, math)\n\n'
    "Rules:\n"
    "- AI/ML in general learning context (even philosophical) → learning\n"
    "- Mentions 'interview', 'mock', 'prepare for', '面试', '考我', '模拟' → interview\n"
    "- Multiple intents → pick the first/primary intent\n"
    "- Short follow-ups (e.g., 'continue', 'go on') → classify based on conversation context\n\n"
    "Output ONLY this JSON, no other text:\n"
    "{{\n"
    '  "main_intent": "learning|resume|interview|job_search|out_of_scope",\n'
    '  "confidence": 0.0,\n'
    '  "needs_clarification": false,\n'
    '  "clarify_question": null,\n'
    '  "reason": "brief reason"\n'
    "}}\n\n"
    "Recent context:\n{context}\n\n"
    "Query: {query}"
)

# 重试时使用更严格的 prompt
_RETRY_PROMPT = ChatPromptTemplate.from_template(
    "Output ONLY valid JSON. No explanation, no markdown.\n\n"
    'Format: {{"main_intent":"learning|resume|interview|job_search|out_of_scope",'
    '"confidence":0.0,"needs_clarification":false,"clarify_question":null,"reason":"string"}}\n\n'
    "Query: {query}\n\nJSON:"
)


# ── 解析工具函数 ───────────────────────────────────────────────────────────────

def _fallback_decision(reason: str = "parse_error") -> RoutingDecision:
    return {
        "main_intent": "out_of_scope",
        "sub_intent": "null",
        "confidence": 0.0,
        "needs_clarification": True,
        "clarify_question": "您能描述一下您想要什么帮助吗？",
        "reason": reason,
    }


def _parse_routing_decision(raw: str, user_text: str, attempt: int = 1) -> RoutingDecision:
    """解析 LLM 输出的 JSON，失败时重试一次，最终兜底到澄清分支。"""
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return _retry_or_fallback(user_text, attempt, "json_not_found")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return _retry_or_fallback(user_text, attempt, "json_parse_error")

    if data.get("main_intent") not in VALID_MAIN:
        return _retry_or_fallback(user_text, attempt, "invalid_enum")

    return {
        "main_intent": data["main_intent"],
        "sub_intent": data.get("sub_intent", "null"),
        "confidence": float(data.get("confidence", 0.8)),
        "needs_clarification": bool(data.get("needs_clarification", False)),
        "clarify_question": data.get("clarify_question"),
        "reason": data.get("reason", ""),
    }


def _retry_or_fallback(user_text: str, attempt: int, error_reason: str) -> RoutingDecision:
    """枚举/解析失败时，对原始输入重试一次；两次均失败则返回澄清兜底。"""
    if attempt < 2:
        print(f"⚠️ 路由解析失败({error_reason})，正在重试...")
        raw = (_RETRY_PROMPT | llm).invoke({"query": user_text}).content
        return _parse_routing_decision(raw, user_text, attempt=2)
    print(f"⚠️ 路由解析二次失败({error_reason})，走澄清兜底")
    return _fallback_decision(reason=error_reason)


# ── 节点函数 ───────────────────────────────────────────────────────────────────

def categorize(state: State, config: RunnableConfig) -> dict:
    """将用户查询主分类，输出结构化 RoutingDecision。

    双写兼容：同时更新 routing_decision（新）和 category（过渡期保留）。
    进入节点时若 state.user_profile 为空，则从 user_store 加载跨会话画像。
    """
    user_text = get_latest_user_text(state)

    recent_msgs = state.get("messages", [])[:-1][-4:]
    context_parts = []
    for msg in recent_msgs:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        context_parts.append(f"{role}: {msg.content[:200]}")
    recent_context = "\n".join(context_parts)

    print("正在对用户问题进行主分类...")
    raw = (_CATEGORIZE_PROMPT | llm).invoke({"query": user_text, "context": recent_context}).content
    decision = _parse_routing_decision(raw, user_text)

    update = {
        "routing_decision": decision,
        "category": decision["main_intent"],  # 双写过渡
        "clarify_count": state.get("clarify_count", 0) if decision["needs_clarification"] is False else state.get("clarify_count", 0),
    }

    # 跨会话画像加载：首次进入该会话时注入用户画像
    if not state.get("user_profile"):
        user_id = (config or {}).get("configurable", {}).get("user_id") or ""
        if user_id:
            persisted_profile = user_store.get_profile(user_id)
            if persisted_profile:
                print(f"[Profile] 从 user_store 加载用户 {user_id} 的画像")
                update["user_profile"] = persisted_profile

    return update


def handle_learning_resource(state: State) -> dict:
    """将学习类查询细分为 tutorial 或 question，更新 routing_decision.sub_intent。"""
    user_text = get_latest_user_text(state)

    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these sub-categories:\n\n"
        "- tutorial: creating tutorials, blogs, documentation, guides on generative AI\n"
        "- question: general questions about generative AI topics\n"
        "Default to question if unclear.\n\n"
        "Output ONLY one word: tutorial or question\n\n"
        "Query: {query}"
    )

    print("正在对学习类问题进行进一步细分...")
    response = (prompt | llm).invoke({"query": user_text}).content.strip().lower()
    sub = "tutorial" if "tutorial" in response else "question"

    rd = dict(state.get("routing_decision") or {})
    rd["sub_intent"] = sub
    return {
        "routing_decision": rd,
        "category": response,  # 双写过渡
    }


def handle_interview_preparation(state: State) -> dict:
    """将面试类查询细分为 mock 或 question，更新 routing_decision.sub_intent。"""
    user_text = get_latest_user_text(state)

    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these sub-categories:\n\n"
        "- mock: requests for mock interviews or practice sessions\n"
        "- question: general interview questions or preparation advice\n"
        "Default to question if unclear.\n\n"
        "Output ONLY one word: mock or question\n\n"
        "Query: {query}"
    )

    print("正在对面试类问题进行进一步细分...")
    response = (prompt | llm).invoke({"query": user_text}).content.strip().lower()
    sub = "mock" if "mock" in response else "question"

    rd = dict(state.get("routing_decision") or {})
    rd["sub_intent"] = sub
    return {
        "routing_decision": rd,
        "category": response,  # 双写过渡
    }


def out_of_scope(state: State) -> dict:
    """处理不受支持的查询，返回友好提示作为 AIMessage。"""
    msg = (
        "抱歉，作为 GenAI 职业助手，我暂时不支持该功能。\n"
        "目前我仅支持以下四个方向：\n"
        "  1. 📚 学习生成式 AI (教程与问答)\n"
        "  2. 📄 简历制作与评审\n"
        "  3. 🎯 面试准备 (题目与模拟面试)\n"
        "  4. 🔍 求职辅助\n"
        "您可以重新输入上述相关的请求。"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "response": msg,
    }
