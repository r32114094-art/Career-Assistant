"""
agents/interview_agent.py - 面试 Agent

包含 InterviewAgent 类，负责：
- Interview_questions: 单轮面试题目生成/互动（AgentExecutor 闭环）
- Mock_Interview:      单轮模拟面试回复（直接 LLM 调用）
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from config import llm
from utils import trim_conversation


class InterviewAgent:
    """处理面试准备相关任务的 Agent。"""

    def __init__(self, prompt=None):
        """
        Args:
            prompt: ChatPromptTemplate（Interview_questions 使用）或 None（Mock_Interview 不需要）
        """
        self.model = llm
        self.prompt = prompt
        search = SerpAPIWrapper()
        self.tools = [
            Tool(
                name="Google_Search",
                func=search.run,
                description="当需要从互联网上搜索最新资讯、面试经验或技术规范时使用这个工具"
            )
        ]

    def Interview_questions(self, user_input: str, chat_history: list = None) -> str:
        """单轮面试题目生成/互动。

        AgentExecutor 内部闭环处理搜索与问题生成，每次调用只做一轮交互。

        Args:
            user_input:   当前用户输入
            chat_history: 之前的对话历史消息列表

        Returns:
            str: LLM 的回复文本
        """
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history or []
        })
        return str(response.get("output", ""))

    def Mock_Interview(self, messages: list) -> str:
        """单轮模拟面试：接收完整对话历史（含系统提示），返回面试官的回复。

        多轮面试对话由外层图的 Checkpointer 驱动，本方法每次只做一轮。

        Args:
            messages: 完整的消息列表（SystemMessage + 历史对话）

        Returns:
            str: 面试官的回复文本
        """
        trimmed = trim_conversation(messages)
        response = self.model.invoke(trimmed)
        return response.content
