"""
nodes/resume.py - 简历制作节点函数

包含一个 LangGraph 节点函数：
- handle_resume_making: 启动简历制作对话
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from state import State
from utils import show_md_file, get_current_time
from agents.resume_agent import ResumeMaker


def handle_resume_making(state: State) -> State:
    """通过多轮对话生成定制化 AI 工程师简历。"""
    system_message = (
        "You are a skilled resume expert with extensive experience in crafting resumes "
        "tailored for tech roles, especially in AI and Generative AI. "
        "Your task is to create a resume template for an AI Engineer specializing in "
        "Generative AI, incorporating trending keywords and technologies in the current "
        "job market. "
        "Feel free to ask users for any necessary details such as skills, experience, "
        "or projects to complete the resume. "
        "Try to ask details step by step and try to ask all details within 4 to 5 steps. "
        "Ensure the final resume is in .md format.\n\n"
        f"[重要环境变量：当前物理时间为 {get_current_time()}。请用此时间作为推断候选人项目经验的最高年份基准。]"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    resume_maker = ResumeMaker(prompt)
    path = resume_maker.Create_Resume(state["query"])
    show_md_file(path)
    return {"response": path}
