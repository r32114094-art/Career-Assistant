"""
agents/resume_agent.py - 简历完善 Agent

包含 ResumeImprover 类，通过 AgentExecutor 分析用户已有简历并给出改进建议。
若 state 中存有求职搜索结果（pending_job_results），则针对具体 JD 给出定制化建议。
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from config import llm_pro


class ResumeImprover:
    """分析用户简历并提供改进建议的 Agent（单轮模式）。"""

    def __init__(self, prompt):
        self.model = llm_pro
        self.prompt = prompt
        search = SerpAPIWrapper()
        self.tools = [
            Tool(
                name="Google_Search",
                func=search.run,
                description="搜索最新的简历标准、行业关键词、ATS 系统要求等",
            )
        ]
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def Improve_Resume(self, user_input: str, chat_history: list = None) -> str:
        """单轮简历改进对话。

        Agent 根据上下文（简历内容 + 可选 JD）决定是追问还是给出改进建议。

        Args:
            user_input:   当前用户输入（含简历文本或追问回复）
            chat_history: 历史对话消息列表

        Returns:
            str: 改进建议或追问文本
        """
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history or [],
        })
        return str(response.get("output", ""))
