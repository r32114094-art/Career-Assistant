"""
state.py - 状态定义模块

使用 TypedDict + LangGraph 的 add_messages Reducer 定义工作流中共享的状态结构。
messages 字段通过 Reducer 自动将新消息追加合并到历史中，是多轮对话的核心支撑。
"""
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


class RoutingDecision(TypedDict):
    """分类节点输出的结构化路由决策。

    Attributes:
        main_intent: 主分类枚举值 learning|resume|interview|job_search|out_of_scope
        sub_intent:  子分类枚举值 question|tutorial|mock|null（由子分类节点填写）
        confidence:  模型自报置信度 0.0~1.0
        needs_clarification: 是否需要向用户澄清意图
        clarify_question: 澄清问题文本，needs_clarification=True 时必填
        reason:      路由原因，用于日志与误差分桶
    """
    main_intent: str
    sub_intent: str
    confidence: float
    needs_clarification: bool
    clarify_question: Optional[str]
    reason: str


class State(TypedDict):
    """LangGraph 工作流的状态结构

    Attributes:
        messages: 完整的对话消息列表，使用 add_messages Reducer 自动合并新旧消息
        category: LLM 分类后的类别标签（过渡期双写保留，后续可移除）
        routing_decision: 结构化路由决策对象，取代 category 成为路由依据
        clarify_count: 已向用户发出澄清问题的轮次，上限 2 次防止无限追问
        response: 最终响应的文件路径或内容
        pending_job_results: 求职搜索完成后的原始 Markdown 文本，
                             暂存供 job_search_review 节点做 HITL 审核用，
                             审核通过后才写入文件。
    """
    messages: Annotated[list, add_messages]
    category: str
    routing_decision: RoutingDecision
    clarify_count: int
    response: str
    pending_job_results: str


def get_latest_user_text(state: State) -> str:
    """从 State 的 messages 中提取最后一条用户消息的文本内容。

    供所有节点函数调用，避免重复编写提取逻辑。

    Args:
        state: 当前工作流状态

    Returns:
        str: 最后一条 HumanMessage 的文本，如果没有则返回空字符串
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def get_chat_history(state: State) -> list:
    """提取除最后一条消息外的所有历史消息，供 AgentExecutor 的 chat_history 参数使用。

    Args:
        state: 当前工作流状态

    Returns:
        list: 历史消息列表（不含最新的用户输入）
    """
    msgs = state.get("messages", [])
    return list(msgs[:-1]) if len(msgs) > 1 else []

