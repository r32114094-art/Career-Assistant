"""
nodes/job_search.py - 求职搜索节点函数

包含一个 LangGraph 节点函数：
- job_search: 通过搜索引擎查找并整理职位信息
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from state import State
from utils import show_md_file, get_current_time
from agents.job_search_agent import JobSearch


def job_search(state: State) -> State:
    """基于用户查询搜索职位并整理为 Markdown。"""
    system_text = (
        "You are an intelligent Job Search Assistant specializing in Generative AI roles. "
        "Before executing any Google Search, you MUST ensure you have verified the user's desired **Job Role** and **Location** (e.g. city). "
        "If either is missing or vague, reply conversationally to ask the user to provide them. "
        "Once you have confirmed both the Role and Location, use the Google_Search tool to find relevant job openings. "
        "After successfully searching, format the job listings clearly into a Markdown document, making it easy to read. "
        f"\n[重要环境变量：当前物理时间为 {get_current_time()}。在处理职位信息时，请对陈旧的招聘信息进行过滤。]\n\n"
        "IMPORTANT: When you have finished outputting the final Markdown markdown document containing the jobs, you MUST append the exact string '[TASK_DONE]' at the very end of your final response to signal the system to stop and save the file."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    job_search_agent = JobSearch(prompt)
    path = job_search_agent.find_jobs(state["query"])
    show_md_file(path)
    return {"response": path}
