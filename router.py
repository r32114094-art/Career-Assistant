"""
router.py - 条件路由函数

包含三个路由函数，用于 LangGraph 的 add_conditional_edges：
- route_query:     根据主分类（1-4）路由到不同子节点
- route_learning:  根据学习子分类路由到 Q&A / Tutorial
- route_interview: 根据面试子分类路由到 Mock / Question
"""
from state import State


def route_query(state: State):
    """根据主分类编号路由到对应的处理节点。

    Returns:
        str | False: 目标节点名称，或 False（无法匹配时）
    """
    if "1" in state["category"]:
        print("类别: handle_learning_resource")
        return "handle_learning_resource"
    elif "2" in state["category"]:
        print("类别: handle_resume_making")
        return "handle_resume_making"
    elif "3" in state["category"]:
        print("类别: handle_interview_preparation")
        return "handle_interview_preparation"
    elif "4" in state["category"]:
        print("类别: job_search")
        return "job_search"
    elif "5" in state["category"]:
        print("类别: out_of_scope")
        return "out_of_scope"
    else:
        print("请根据我的描述提出你的问题。")
        return False


def route_interview(state: State) -> str:
    """根据面试子分类路由到面试题目或模拟面试节点。

    Returns:
        str: 目标节点名称
    """
    if "question" in state["category"].lower():
        print("类别: interview_topics_questions")
        return "interview_topics_questions"
    elif "mock" in state["category"].lower():
        print("类别: mock_interview")
        return "mock_interview"
    else:
        print("类别: mock_interview (default)")
        return "mock_interview"


def route_learning(state: State):
    """根据学习子分类路由到 Q&A 机器人或教程生成节点。

    Returns:
        str | False: 目标节点名称，或 False（无法匹配时）
    """
    if "question" in state["category"].lower():
        print("类别: ask_query_bot")
        return "ask_query_bot"
    elif "tutorial" in state["category"].lower():
        print("类别: tutorial_agent")
        return "tutorial_agent"
    else:
        print("请根据面试相关的描述提出你的问题。")
        return False
