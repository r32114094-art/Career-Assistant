"""
agents/learning_agent.py - 学习资源 Agent

包含 LearningResourceAgent 类，负责：
- TutorialAgent: 基于搜索工具生成教程博客（单次执行，AgentExecutor 闭环）
- QueryBot:      单轮 Q&A 回答（无循环，多轮由图的反复 invoke 驱动）
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from config import llm_pro
from utils import trim_conversation, save_file


class LearningResourceAgent:
    """处理学习资源相关任务的 Agent。"""

    def __init__(self, prompt=None):
        """
        Args:
            prompt: ChatPromptTemplate（TutorialAgent 使用）或 None（QueryBot 不需要）
        """
        self.model = llm_pro
        self.prompt = prompt
        search = SerpAPIWrapper()
        self.tools = [
            Tool(
                name="Google_Search",
                func=search.run,
                description="当需要从互联网上搜索最新技术教程、博客文章时使用这个工具"
            )
        ]

    def TutorialAgent(self, user_input: str) -> str:
        """生成 GenAI 主题的教程博客，并保存为 Markdown 文件。

        单次执行：AgentExecutor 内部闭环处理搜索与生成，无需外部循环。

        Args:
            user_input: 用户查询内容

        Returns:
            str: 保存文件的路径
        """
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})

        content = str(response.get("output")).replace("```markdown", "").strip()
        path = save_file(content, "Tutorial")
        print(f"教程已保存至: {path}")
        return path

    def QueryBot(self, messages: list) -> str:
        """单轮 Q&A 回答：接收完整对话历史（含系统提示），调用 LLM 一次并返回。

        多轮对话由外层图的 Checkpointer 驱动，本方法每次只做一轮问答。

        Args:
            messages: 完整的消息列表（SystemMessage + 历史 HumanMessage/AIMessage）

        Returns:
            str: LLM 的回复文本
        """
        trimmed = trim_conversation(messages)
        response = self.model.invoke(trimmed)
        return response.content
