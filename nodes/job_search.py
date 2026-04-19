"""
nodes/job_search.py - 求职搜索节点函数

包含一个 LangGraph 节点函数：
- job_search: 调用 Agent 搜索职位，将结果暂存入 state["pending_job_results"]，
              由后续的 job_search_review 节点负责 HITL 审核后再写入文件。

【核心设计原则】
  本节点无 interrupt()，只负责"搜索"这一个有副作用的动作。
  搜索结果暂存于 state["pending_job_results"]，不立即写文件。
  这样 HITL resume 时只有 job_search_review 节点（无副作用）重放，
  不会重复触发 API 搜索。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from state import State, get_latest_user_text, get_chat_history, format_profile_context
from utils import get_current_time
from agents.job_search_agent import JobSearch


def job_search(state: State) -> State:
    """调用 Agent 搜索职位，将格式化结果存入 state["pending_job_results"]。

    不做 interrupt()，也不写文件，使 HITL resume 时的节点重放完全安全。
    """
    profile_ctx = format_profile_context(state.get("user_profile") or {})
    system_text = (
        "You are an intelligent Job Search Assistant specializing in Generative AI roles. "
        "You have access to four tools — use them selectively based on user intent:\n"
        "- search_jobs: 搜索职位列表。调用前必须确认用户的 Job Role 和 Location；若缺失则先追问。\n"
        "- search_salary: 用户询问薪资待遇或想了解市场行情时使用。\n"
        "- search_company: 用户想了解某家具体公司的背景、文化或评价时使用。\n"
        "- analyze_job_fit: 用户提供了个人背景并想评估与某职位的匹配度时使用；"
        "需从对话中提取 job_description 和 user_background 两个参数。\n\n"
        "搜索到职位后，以 Markdown 格式整理输出，过滤明显过时的信息。\n"
        + (f"{profile_ctx}\n" if profile_ctx else "")
        + f"[当前时间：{get_current_time()}]\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    user_text = get_latest_user_text(state)
    chat_history = get_chat_history(state)

    job_search_agent = JobSearch(prompt)
    response_text = job_search_agent.find_jobs(user_text, chat_history)

    # 如果结果包含 Markdown 标志，视为真实搜索结果，暂存待下游审核节点处理
    if "```" in response_text or "##" in response_text:
        clean_text = response_text.replace("```markdown", "").replace("```", "").strip()
        return {
            "messages": [AIMessage(content="🔍 职位搜索完成，正在整理结果，请稍后确认...")],
            "pending_job_results": clean_text,
            "response": "",
        }

    # 普通对话（Agent 正在追问用户需求），直接返回
    return {
        "messages": [AIMessage(content=response_text)],
        "pending_job_results": "",
        "response": response_text,
    }
