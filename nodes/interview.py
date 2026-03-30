"""
nodes/interview.py - 面试类节点函数

包含两个 LangGraph 节点函数：
- interview_topics_questions: 面试题目问答
- mock_interview:             模拟面试
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from state import State
from utils import show_md_file, get_current_time
from agents.interview_agent import InterviewAgent


def interview_topics_questions(state: State) -> State:
    """提供定制化的 GenAI 面试题目列表。"""
    system_message = (
        "You are a good researcher in finding interview questions for Generative AI topics and jobs. "
        "Your task is to provide a list of interview questions for Generative AI topics and job "
        "based on user requirements. "
        "Provide top questions with references and links if possible. "
        "You may ask for clarification if needed. "
        "Generate a .md document containing the questions."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。你在搜索面经或题库时，务必加上此年份作为过滤条件，不要使用知识库停滞期的旧时间。]"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}


def mock_interview(state: State) -> State:
    """启动 GenAI 模拟面试会话，结束时给出评估。"""
    system_message = (
        "You are a Generative AI Interviewer. You have conducted numerous interviews "
        "for Generative AI roles. "
        "Your task is to conduct a mock interview for a Generative AI position, "
        "engaging in a back-and-forth interview session. "
        "The conversation should not exceed more than 15 to 20 minutes. "
        "At the end of the interview, provide an evaluation for the candidate."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。如果在模拟面试中涉及最新的前沿技术，请以此物理时间为发展基线。]"
    prompt = [SystemMessage(content=system_message)]

    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Mock_Interview()
    show_md_file(path)
    return {"response": path}
