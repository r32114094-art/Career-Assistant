"""
nodes/learning.py - 学习类节点函数

包含两个 LangGraph 节点函数：
- ask_query_bot:  启动 Q&A 对话会话
- tutorial_agent:  生成教程博客
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from state import State
from utils import show_md_file, get_current_time
from agents.learning_agent import LearningResourceAgent


def ask_query_bot(state: State) -> State:
    """启动 GenAI 专家 Q&A 对话会话。"""
    system_message = (
        "You are an expert Generative AI Engineer with extensive experience "
        "in training and guiding others in AI engineering. "
        "You have a strong track record of solving complex problems and "
        "addressing various challenges in AI. "
        "Your role is to assist users by providing insightful solutions "
        "and expert advice on their queries. "
        "Engage in a back-and-forth chat session to address user queries."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。请确保你的技术解答符合这个时代的前沿情况。]"
    prompt = [SystemMessage(content=system_message)]

    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.QueryBot(state["query"])
    show_md_file(path)
    return {"response": path}


def tutorial_agent(state: State) -> State:
    """生成 GenAI 主题的教程博客并保存。"""
    system_message = (
        "You are a knowledgeable assistant specializing as a Senior Generative AI Developer "
        "with extensive experience in both development and tutoring. "
        "Additionally, you are an experienced blogger who creates tutorials focused on Generative AI. "
        "Your task is to develop high-quality tutorials blogs in .md file with Coding example "
        "based on the user's requirements. "
        "Ensure tutorial includes clear explanations, well-structured python code, comments, "
        "and fully functional code examples. "
        "Provide resource reference links at the end of each tutorial for further learning."
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。你在编写教程和搜集技术标准时，若有明确的时效性，请以当下的时间为准进行检索。]"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(state["query"])
    show_md_file(path)
    return {"response": path}
