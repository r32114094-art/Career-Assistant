"""
nodes/learning.py - 学习类节点函数

包含两个 LangGraph 节点函数：
- ask_query_bot:   单轮 Q&A 回答，返回 AIMessage
- tutorial_agent:  生成教程博客，保存文件并返回 AIMessage
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage

from state import State, get_latest_user_text, format_profile_context
from utils import show_md_file, get_current_time
from agents.learning_agent import LearningResourceAgent


def ask_query_bot(state: State) -> State:
    """单轮 GenAI 专家 Q&A：将系统提示与历史消息合并，调用 LLM 一次。"""
    profile_ctx = format_profile_context(state.get("user_profile") or {})
    system_message = (
        "You are an expert Generative AI Engineer with extensive experience "
        "in training and guiding others in AI engineering. "
        "You have a strong track record of solving complex problems and "
        "addressing various challenges in AI. "
        "Your role is to assist users by providing insightful solutions "
        "and expert advice on their queries. "
        "Engage in a back-and-forth chat session to address user queries."
        + (f"\n\n{profile_ctx}" if profile_ctx else "")
    )
    system_message += f"\n\n[重要环境变量：当前物理时间为 {get_current_time()}。请确保你的技术解答符合这个时代的前沿情况。]"

    # 构建完整消息列表：系统提示 + 历史对话
    full_messages = [SystemMessage(content=system_message)] + list(state["messages"])

    learning_agent = LearningResourceAgent()
    response_text = learning_agent.QueryBot(full_messages)

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
    }


def tutorial_agent(state: State) -> State:
    """生成 GenAI 主题的教程并保存为文件。"""
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

    user_text = get_latest_user_text(state)
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(user_text)
    show_md_file(path)

    return {
        "messages": [AIMessage(content=f"教程已生成并保存至: {path}")],
        "response": path,
    }
