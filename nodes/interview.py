"""
nodes/interview.py - 面试类节点函数

包含两个 LangGraph 节点函数：
- interview_topics_questions: 单轮面试题目生成/互动
- mock_interview:             单轮模拟面试回复
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage

from state import State, get_latest_user_text, get_chat_history
from utils import get_current_time
from agents.interview_agent import InterviewAgent


def interview_topics_questions(state: State) -> State:
    """单轮面试题目生成/互动，通过 AgentExecutor 闭环处理。"""
    system_message = (
        "You are a good researcher in finding interview questions for Generative AI topics and jobs. "
        "Your task is to provide a list of interview questions for Generative AI topics and job "
        "based on user requirements. "
        "Provide top questions with references and links if possible. "
        "You may ask for clarification if needed. "
        "Generate a .md document containing the questions."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。你在搜索面经或题库时，务必加上此年份作为过滤条件。]"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    user_text = get_latest_user_text(state)
    chat_history = get_chat_history(state)

    interview_agent = InterviewAgent(prompt)
    response_text = interview_agent.Interview_questions(user_text, chat_history)

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
    }


def mock_interview(state: State) -> State:
    """单轮模拟面试：将系统提示与历史消息合并，面试官做一次回复。"""
    system_message = (
        "You are a Generative AI Interviewer. You have conducted numerous interviews "
        "for Generative AI roles. "
        "Your task is to conduct a mock interview for a Generative AI position, "
        "engaging in a back-and-forth interview session. "
        "The conversation should not exceed more than 15 to 20 minutes. "
        "At the end of the interview, provide an evaluation for the candidate."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。如果在模拟面试中涉及最新的前沿技术，请以此物理时间为发展基线。]"

    # 构建完整消息列表：系统提示 + 历史对话
    full_messages = [SystemMessage(content=system_message)] + list(state["messages"])

    interview_agent = InterviewAgent()
    response_text = interview_agent.Mock_Interview(full_messages)

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
    }
