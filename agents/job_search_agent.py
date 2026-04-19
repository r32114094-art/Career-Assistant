"""
agents/job_search_agent.py - 求职搜索 Agent

包含 JobSearch 类，利用多个工具完成求职相关任务：
- search_jobs:      搜索职位列表（需确认 role + location）
- search_salary:    查询薪资行情
- search_company:   搜索目标公司背景与文化
- analyze_job_fit:  分析用户背景与 JD 的匹配度（LLM 工具，无外部 API）
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor

from config import llm_pro, llm
from utils import get_current_time


def _make_tools(search: SerpAPIWrapper, current_time: str) -> list:
    jobs_search = SerpAPIWrapper(params={"engine": "google_jobs", "hl": "zh-cn", "gl": "cn"})

    @tool
    def search_jobs(role: str, location: str, job_type: str = "full-time") -> str:
        """搜索特定职位和城市的招聘信息。需要提供职位名称和城市，job_type 可选。"""
        query = f"{role} {location}"
        if job_type and job_type != "full-time":
            query += f" {job_type}"
        try:
            results = jobs_search.results(query)
            jobs = results.get("jobs_results", [])
            if not jobs:
                return search.run(f"{role} {location} 招聘 {current_time}")
            lines = []
            for j in jobs[:8]:
                extensions = j.get("extensions", [])
                extras = " | ".join(extensions) if extensions else ""
                lines.append(
                    f"【{j.get('title', '')}】{j.get('company_name', '')} · {j.get('location', '')}\n"
                    f"  {extras}\n"
                    f"  {j.get('description', '')[:120]}..."
                )
            return f"找到 {len(jobs)} 个职位（显示前8条）：\n\n" + "\n\n".join(lines)
        except Exception:
            return search.run(f"{role} {location} 招聘 {current_time}")

    @tool
    def search_salary(role: str, location: str) -> str:
        """查询特定职位在指定城市的薪资范围和市场行情。"""
        return search.run(f"{role} {location} 薪资范围 平均工资 薪酬 {current_time}")

    @tool
    def search_company(company_name: str) -> str:
        """搜索公司背景、企业文化和员工评价，帮助用户评估目标雇主。"""
        return search.run(f"{company_name} 公司介绍 企业文化 员工评价 工作氛围")

    @tool
    def analyze_job_fit(job_description: str, user_background: str) -> str:
        """根据用户背景和职位描述分析匹配度与技能差距。job_description 填 JD 内容，user_background 填用户描述的技能和经历。"""
        prompt = (
            "请分析以下用户背景与职位要求的匹配程度，简洁输出：\n\n"
            f"【用户背景】\n{user_background}\n\n"
            f"【职位描述】\n{job_description}\n\n"
            "输出格式：\n"
            "1. 匹配度评分（0-100）\n"
            "2. 核心优势（3条）\n"
            "3. 技能差距（2-3条）\n"
            "4. 提升建议（1-2条）"
        )
        return llm.invoke([HumanMessage(content=prompt)]).content

    return [search_jobs, search_salary, search_company, analyze_job_fit]


class JobSearch:
    """通过多工具查找职位、分析薪资、调研公司、评估匹配度的 Agent。"""

    def __init__(self, prompt):
        """
        Args:
            prompt: ChatPromptTemplate，需包含 chat_history / input / agent_scratchpad 占位符
        """
        self.model = llm_pro
        self.prompt = prompt
        self.tools = _make_tools(SerpAPIWrapper(), get_current_time())
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def find_jobs(self, user_input: str, chat_history: list = None) -> str:
        """单轮求职对话/搜索。

        AgentExecutor 将根据上下文决定是追问用户（缺少职位或地点）还是执行搜索。

        Args:
            user_input:   当前用户输入
            chat_history: 之前的对话历史消息列表

        Returns:
            str: LLM 的回复文本（可能是追问或搜索结果）
        """
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history or []
        })
        return str(response.get("output", ""))
