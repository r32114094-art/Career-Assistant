"""
agents/job_search_agent.py - 求职搜索 Agent

包含 JobSearch 类，利用搜索工具查找市场职位。
单轮模式：AgentExecutor 根据上下文决定是追问用户还是执行搜索。
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from config import llm_pro


class JobSearch:
    """通过搜索工具查找并整理 GenAI 相关职位的 Agent。"""

    def __init__(self, prompt):
        """
        Args:
            prompt: ChatPromptTemplate，需包含 chat_history / input / agent_scratchpad 占位符
        """
        self.model = llm_pro
        self.prompt = prompt
        search = SerpAPIWrapper()
        self.tools = [
            Tool(
                name="Google_Search",
                func=search.run,
                description="专门用于搜索特定职位、工作机会或行业的招聘信息"
            )
        ]
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
