"""
agents/resume_agent.py - 简历制作 Agent

包含 ResumeMaker 类，通过单次 AgentExecutor 调用处理简历制作的一轮对话。
多轮信息收集由外层图的 Checkpointer 驱动。
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from config import llm_pro


class ResumeMaker:
    """通过 AgentExecutor 创建专业简历的 Agent（单轮模式）。"""

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
                description="当需要搜索互联网上最新的简历标准和写法时使用这个工具"
            )
        ]
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def Create_Resume(self, user_input: str, chat_history: list = None) -> str:
        """单轮简历制作对话。

        AgentExecutor 将根据上下文决定是追问用户信息，还是生成最终简历。

        Args:
            user_input:   当前用户输入
            chat_history: 之前的对话历史消息列表

        Returns:
            str: LLM 的回复文本（可能是追问或最终简历内容）
        """
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history or []
        })
        return str(response.get("output", ""))
