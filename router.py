"""
router.py - 条件路由函数

包含四个路由函数，用于 LangGraph 的 add_conditional_edges：
- route_query:     读取 routing_decision.main_intent 路由到不同子节点，
                   低置信度或需澄清时转到 clarify 节点
- route_learning:  读取 routing_decision.sub_intent 路由到 Q&A / Tutorial
- route_interview: 读取 routing_decision.sub_intent 路由到 Mock / Question
- route_job_search: 根据 pending_job_results 决定是否进入 HITL 审核节点
"""
from state import State

CONFIDENCE_THRESHOLD = 0.65


def route_query(state: State) -> str:
    """根据结构化路由决策路由到对应处理节点。

    优先检查置信度与澄清标志：
    - needs_clarification=True 或 confidence < 0.65 且澄清次数未超上限 → clarify
    - 否则按 main_intent 字段映射到目标节点

    Returns:
        str: 目标节点名称
    """
    rd = state.get("routing_decision") or {}
    needs_clarification = rd.get("needs_clarification", False)
    confidence = rd.get("confidence", 1.0)
    clarify_count = state.get("clarify_count", 0)

    if (needs_clarification or confidence < CONFIDENCE_THRESHOLD) and clarify_count < 2:
        print(f"⚠️ 置信度={confidence:.2f} 或需澄清，路由至 clarify（第{clarify_count + 1}次）")
        return "clarify"

    main_intent = rd.get("main_intent", "out_of_scope")
    mapping = {
        "learning": "handle_learning_resource",
        "resume": "handle_resume_improvement",
        "interview": "handle_interview_preparation",
        "job_search": "job_search",
        "out_of_scope": "out_of_scope",
    }
    target = mapping.get(main_intent, "out_of_scope")
    print(f"类别: {target} (confidence={confidence:.2f}, reason={rd.get('reason', '')})")
    return target


def route_interview(state: State) -> str:
    """根据 routing_decision.sub_intent 路由到面试题目或模拟面试节点。

    Returns:
        str: 目标节点名称
    """
    rd = state.get("routing_decision") or {}
    sub = rd.get("sub_intent", "question")
    if sub == "mock":
        print("类别: mock_interview")
        return "mock_interview"
    print("类别: interview_topics_questions")
    return "interview_topics_questions"


def route_learning(state: State) -> str:
    """根据 routing_decision.sub_intent 路由到 Q&A 机器人或教程生成节点。

    Returns:
        str: 目标节点名称
    """
    rd = state.get("routing_decision") or {}
    sub = rd.get("sub_intent", "question")
    if sub == "tutorial":
        print("类别: tutorial_agent")
        return "tutorial_agent"
    print("类别: ask_query_bot")
    return "ask_query_bot"


def route_job_search(state: State) -> str:
    """根据 pending_job_results 决定是否进入 HITL 审核节点。

    有搜索结果 → job_search_review（触发 interrupt() 等待用户确认）
    无搜索结果 → end（本轮只是 Agent 追问对话，直接结束）
    """
    if state.get("pending_job_results"):
        print("类别: job_search_review (有待审核结果)")
        return "job_search_review"
    print("类别: end (无搜索结果，普通对话)")
    return "end"
