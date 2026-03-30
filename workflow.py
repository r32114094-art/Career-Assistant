"""
workflow.py - LangGraph 工作流定义

构建并编译 GenAI Career Assistant 的状态图工作流：
1. 添加所有节点
2. 定义起始边和条件边
3. 定义终止边
4. 编译为可执行应用
"""
from langgraph.graph import StateGraph, END, START

from state import State
from nodes import (
    categorize,
    handle_learning_resource,
    handle_interview_preparation,
    ask_query_bot,
    tutorial_agent,
    interview_topics_questions,
    mock_interview,
    handle_resume_making,
    job_search,
    out_of_scope,
)
from router import route_query, route_interview, route_learning


def build_workflow():
    """构建并编译 LangGraph 工作流。

    Returns:
        CompiledGraph: 可调用的工作流应用
    """
    workflow = StateGraph(State)

    # ── 添加节点 ──────────────────────────────────────────
    workflow.add_node("categorize", categorize)
    workflow.add_node("handle_learning_resource", handle_learning_resource)
    workflow.add_node("handle_resume_making", handle_resume_making)
    workflow.add_node("handle_interview_preparation", handle_interview_preparation)
    workflow.add_node("job_search", job_search)
    workflow.add_node("mock_interview", mock_interview)
    workflow.add_node("interview_topics_questions", interview_topics_questions)
    workflow.add_node("tutorial_agent", tutorial_agent)
    workflow.add_node("ask_query_bot", ask_query_bot)
    workflow.add_node("out_of_scope", out_of_scope)

    # ── 起始边：从 START 到分类节点 ──────────────────────
    workflow.add_edge(START, "categorize")

    # ── 主分类条件边 ─────────────────────────────────────
    workflow.add_conditional_edges(
        "categorize",
        route_query,
        {
            "handle_learning_resource": "handle_learning_resource",
            "handle_resume_making": "handle_resume_making",
            "handle_interview_preparation": "handle_interview_preparation",
            "job_search": "job_search",
            "out_of_scope": "out_of_scope",
        },
    )

    # ── 面试子分类条件边 ─────────────────────────────────
    workflow.add_conditional_edges(
        "handle_interview_preparation",
        route_interview,
        {
            "mock_interview": "mock_interview",
            "interview_topics_questions": "interview_topics_questions",
        },
    )

    # ── 学习子分类条件边 ─────────────────────────────────
    workflow.add_conditional_edges(
        "handle_learning_resource",
        route_learning,
        {
            "tutorial_agent": "tutorial_agent",
            "ask_query_bot": "ask_query_bot",
        },
    )

    # ── 终止边 ──────────────────────────────────────────
    workflow.add_edge("handle_resume_making", END)
    workflow.add_edge("job_search", END)
    workflow.add_edge("interview_topics_questions", END)
    workflow.add_edge("mock_interview", END)
    workflow.add_edge("ask_query_bot", END)
    workflow.add_edge("tutorial_agent", END)
    workflow.add_edge("out_of_scope", END)

    # ── 设置入口并编译 ──────────────────────────────────
    workflow.set_entry_point("categorize")
    app = workflow.compile()
    return app
