"""
nodes/categorize.py - 分类节点

包含三个 LangGraph 节点函数：
- categorize:                  将用户查询归类为 1-4 大类
- handle_learning_resource:    将学习类查询细分为 Tutorial / Question
- handle_interview_preparation: 将面试类查询细分为 Mock / Question
"""
from langchain_core.prompts import ChatPromptTemplate

from config import llm  #导入能力较低的deepseekV3
from state import State


def categorize(state: State) -> State:
    """将用户查询分类为四大主类别，返回类别数字（1-5）。

    类别：
        1 - Learn Generative AI Technology
        2 - Resume Making
        3 - Interview Preparation
        4 - Job Search
        5 - Other / Out of Scope
    """
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "5: Other / Out of Scope (e.g., cooking, weather, casual chat)\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> 1\n"
        "2. Query: 'Can you help me improve my resume for a tech position?' -> 2\n"
        "3. Query: 'What are some common questions asked in AI interviews?' -> 3\n"
        "4. Query: 'Are there any job openings for AI engineers?' -> 4\n"
        "5. Query: 'I want to cook some fried rice, give me a recipe.' -> 5\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )  #利用few shots，让模型更好地理解任务，回答输出更准确

    chain = prompt | llm   #LCEL编排
    print("正在对用户问题进行主分类...")
    category = chain.invoke({"query": state["query"]}).content   #把State类型中的query取出来，传递给prompt，然后把prompt传递给llm，最后把llm的输出传递给State类型中的category
    return {"category": category}


def handle_learning_resource(state: State) -> State:
    """将学习类查询细分为 Tutorial 或 Question。"""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.\n"
        "- Question: For general queries asking about generative AI topics.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'How to create a blog on prompt engineering for generative AI?' -> Category: Tutorial\n"
        "2. User query: 'Can you provide a step-by-step guide on fine-tuning a generative model?' -> Category: Tutorial\n"
        "3. User query: 'Provide me the documentation for Langchain?' -> Category: Tutorial\n"
        "4. User query: 'What are the main applications of generative AI?' -> Category: Question\n"
        "5. User query: 'Is there any generative AI course available?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    print("正在对学习类问题进行进一步细分...")
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}


def handle_interview_preparation(state: State) -> State:
    """将面试类查询细分为 Mock 或 Question。"""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Mock: For requests related to mock interviews.\n"
        "- Question: For general queries asking about interview topics or preparation.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'Can you conduct a mock interview with me for a Gen AI role?' -> Category: Mock\n"
        "2. User query: 'What topics should I prepare for an AI Engineer interview?' -> Category: Question\n"
        "3. User query: 'I need to practice interview focused on Gen AI.' -> Category: Mock\n"
        "4. User query: 'Can you list important coding topics for AI tech interviews?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    print("正在对面试类问题进行进一步细分...")
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def out_of_scope(state: State) -> State:
    """处理不受支持的查询。"""
    print("\n抱歉，作为 GenAI 职业助手，我暂时不支持该功能。")
    print("目前我仅支持以下四个方向：")
    print("  1. 学习生成式 AI (教程与问答)")
    print("  2. 制作与评审 AI 简历")
    print("  3. 准备 AI 面试 (题目与模拟)")
    print("  4. 搜索 AI 岗位")
    print("您可以重新输入上述相关的请求。\n")
    return {"response": "Function not supported."}
