"""
router.py - 条件路由函数

包含三个路由函数，用于 LangGraph 的 add_conditional_edges：
- route_query:     根据主分类（1-5）路由到不同子节点
- route_learning:  根据学习子分类路由到 Q&A / Tutorial
- route_interview: 根据面试子分类路由到 Mock / Question

使用正则提取替代原先脆弱的子串匹配，确保路由健壮性。
"""
import re
from state import State


def route_query(state: State) -> str:
    """根据主分类编号路由到对应的处理节点。

    从 category 字段中提取第一个 1-5 的数字，映射到目标节点。
    无法匹配时兜底到 out_of_scope。

    Returns:
        str: 目标节点名称
    """
    category = state.get("category", "").strip()
    match = re.search(r"[1-5]", category)
    if not match:
        print("⚠️ 无法识别分类，路由至 out_of_scope")
        return "out_of_scope"

    num = match.group()
    mapping = {
        "1": "handle_learning_resource",
        "2": "handle_resume_making",
        "3": "handle_interview_preparation",
        "4": "job_search",
        "5": "out_of_scope",
    }
    target = mapping[num]
    print(f"类别: {target}")
    return target


def route_interview(state: State) -> str:
    """根据面试子分类路由到面试题目或模拟面试节点。

    Returns:
        str: 目标节点名称
    """
    category = state.get("category", "").lower().strip()
    if "mock" in category:
        print("类别: mock_interview")
        return "mock_interview"
    else:
        # 默认走面试题目
        print("类别: interview_topics_questions")
        return "interview_topics_questions"


def route_learning(state: State) -> str:
    """根据学习子分类路由到 Q&A 机器人或教程生成节点。

    Returns:
        str: 目标节点名称
    """
    category = state.get("category", "").lower().strip()
    if "tutorial" in category:
        print("类别: tutorial_agent")
        return "tutorial_agent"
    else:
        # 默认走问答
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
