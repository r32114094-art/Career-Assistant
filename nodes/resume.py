"""
nodes/resume.py - 简历完善节点函数

包含一个 LangGraph 节点函数：
- handle_resume_improvement: 分析用户上传的简历内容，给出改进建议。
  用户可在前端同时粘贴岗位 JD，JD 文本随简历内容一起包含在用户消息中，
  节点无需读取 state["pending_job_results"]。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from state import State, get_latest_user_text, get_chat_history, format_profile_context
from utils import get_current_time
from agents.resume_agent import ResumeImprover


def handle_resume_improvement(state: State) -> State:
    """单轮简历完善对话。

    用户消息中可能包含：
    - 【简历内容】 字段：上传解析后的简历全文
    - 【目标岗位 JD】 字段（可选）：用户手动粘贴的岗位描述

    Agent 根据这两部分给出结构化改进建议。
    """
    profile_ctx = format_profile_context(state.get("user_profile") or {})

    system_message = (
        "You are a professional resume coach specializing in AI/ML and Generative AI roles. "
        "Your job is to compare the user's resume against the target job description (if provided) "
        "and give concise, actionable feedback — NOT to rewrite or output the full resume.\n\n"
        "## Output Format (always use these three sections, in Chinese):\n\n"
        "### ✅ 已匹配的技能与亮点\n"
        "List the skills, experiences, and highlights in the resume that directly match "
        "the JD requirements. Quote brief phrases from the resume. "
        "If no JD is provided, list the strongest points for GenAI/ML roles.\n\n"
        "### ❌ 缺少的技能与亮点\n"
        "List skills, keywords, or experiences required or preferred by the JD that are "
        "absent or weak in the resume. Be specific about what is missing and why it matters.\n\n"
        "### 📝 简历修改建议\n"
        "Give 5–8 concrete, prioritized suggestions. For each suggestion:\n"
        "- State what to change and where (which section)\n"
        "- Show a before → after example where applicable\n"
        "- Focus on wording improvements, metric additions, and keyword gaps\n\n"
        "## Hard Rules:\n"
        "- Do NOT output a complete resume or a full .md document\n"
        "- Do NOT rewrite entire sections — only show targeted diffs\n"
        "- Keep each bullet point concise (1–2 lines max)\n"
        "- If no JD is present, note that adding a JD will make suggestions more targeted\n"
        "- If the message has no resume content yet, ask the user to upload their resume\n"
        + (f"\n\n## 用户背景\n{profile_ctx}" if profile_ctx else "")
        + f"\n\n[当前时间：{get_current_time()}]"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    user_text = get_latest_user_text(state)
    chat_history = get_chat_history(state)

    improver = ResumeImprover(prompt)
    response_text = improver.Improve_Resume(user_text, chat_history)

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
    }
