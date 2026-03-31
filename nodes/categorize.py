"""
nodes/categorize.py - 分类节点

包含三个 LangGraph 节点函数：
- categorize:                   将用户查询归类为 1-5 大类（注入历史上下文）
- handle_learning_resource:     将学习类查询细分为 Tutorial / Question
- handle_interview_preparation: 将面试类查询细分为 Mock / Question
- out_of_scope:                 处理不支持的查询
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from config import llm
from state import State, get_latest_user_text


def categorize(state: State) -> State:
    """将用户查询分类为五大主类别，返回类别数字（1-5）。

    注入最近的对话上下文，使路由能够理解多轮对话中的意图延续与切换。

    类别：
        1 - Learn Generative AI Technology
        2 - Resume Making
        3 - Interview Preparation
        4 - Job Search
        5 - Other / Out of Scope
    """
    user_text = get_latest_user_text(state)

    # 从历史消息中构建最近的上下文摘要（最多取最近 4 条）
    recent_context = ""
    recent_msgs = state.get("messages", [])[:-1][-4:]  # 最后 4 条历史消息
    if recent_msgs:
        context_parts = []
        for msg in recent_msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context_parts.append(f"{role}: {msg.content[:200]}")
        recent_context = "\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "5: Other / Out of Scope (e.g., cooking, weather, casual chat)\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI?' -> 1\n"
        "2. Query: 'Can you help me improve my resume?' -> 2\n"
        "3. Query: 'What are common questions in AI interviews?' -> 3\n"
        "4. Query: 'Are there any job openings for AI engineers?' -> 4\n"
        "5. Query: 'I want to cook fried rice.' -> 5\n\n"
        "Recent conversation context (use this to understand ongoing topics):\n"
        "{context}\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )

    chain = prompt | llm
    print("正在对用户问题进行主分类...")
    category = chain.invoke({"query": user_text, "context": recent_context}).content
    return {"category": category.strip()}


def handle_learning_resource(state: State) -> State:
    """将学习类查询细分为 Tutorial 或 Question。"""
    user_text = get_latest_user_text(state)

    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.\n"
        "- Question: For general queries asking about generative AI topics.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. 'How to create a blog on prompt engineering?' -> Category: Tutorial\n"
        "2. 'What are the main applications of generative AI?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    print("正在对学习类问题进行进一步细分...")
    response = chain.invoke({"query": user_text}).content
    return {"category": response.strip()}


def handle_interview_preparation(state: State) -> State:
    """将面试类查询细分为 Mock 或 Question。"""
    user_text = get_latest_user_text(state)

    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Mock: For requests related to mock interviews.\n"
        "- Question: For general queries asking about interview topics or preparation.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. 'Can you conduct a mock interview with me?' -> Category: Mock\n"
        "2. 'What topics should I prepare for an AI interview?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    print("正在对面试类问题进行进一步细分...")
    response = chain.invoke({"query": user_text}).content
    return {"category": response.strip()}


def out_of_scope(state: State) -> State:
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
