"""
workflow.py - LangGraph 工作流定义

构建并编译 GenAI Career Assistant 的状态图工作流：
1. 添加所有节点
2. 定义起始边和条件边
3. 定义终止边
4. 挂载 PostgresSaver 持久化记忆（Supabase PostgreSQL）
5. 编译为可执行应用
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres import PostgresSaver

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
    job_search_review,
    out_of_scope,
)
from router import route_query, route_interview, route_learning, route_job_search


def build_workflow():
    """构建并编译 LangGraph 工作流。

    每次 invoke 的执行路径：
        START → categorize → [条件路由] → 子系统分类(可选) → 叶子节点 → END

    多轮对话由 MemorySaver 驱动：
        - 每次 invoke 携带 thread_id，状态自动恢复
        - 新的 HumanMessage 进入 → 经过路由重新评估意图 → 匹配的叶子节点处理一轮
        - 叶子节点返回 AIMessage 到 state.messages → END
        - 下一次 invoke 再次从 START 开始，但消息历史已自动累积

    Returns:
        CompiledGraph: 可调用的工作流应用（已挂载 Checkpointer）
    """
    workflow = StateGraph(State)

    # ── 添加节点 ──────────────────────────────────────────
    workflow.add_node("categorize", categorize)
    workflow.add_node("handle_learning_resource", handle_learning_resource)
    workflow.add_node("handle_resume_making", handle_resume_making)
    workflow.add_node("handle_interview_preparation", handle_interview_preparation)
    workflow.add_node("job_search", job_search)
    workflow.add_node("job_search_review", job_search_review)
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
    # job_search → job_search_review（有搜索结果）或 END（只是追问对话）
    workflow.add_conditional_edges(
        "job_search",
        route_job_search,
        {
            "job_search_review": "job_search_review",
            "end": END,
        },
    )
    workflow.add_edge("job_search_review", END)
    workflow.add_edge("interview_topics_questions", END)
    workflow.add_edge("mock_interview", END)
    workflow.add_edge("ask_query_bot", END)
    workflow.add_edge("tutorial_agent", END)
    workflow.add_edge("out_of_scope", END)

    # ── 挂载 PostgreSQL 持久化记忆并编译 ────────────────────
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL 环境变量未设置，请检查 .env 文件")

    import psycopg
    conn = psycopg.Connection.connect(
        db_url, autocommit=True, prepare_threshold=0
    )
    postgres_saver = PostgresSaver(conn)
    postgres_saver.setup()  # 首次运行时自动创建 checkpoint 表
    app = workflow.compile(checkpointer=postgres_saver)
    return app
