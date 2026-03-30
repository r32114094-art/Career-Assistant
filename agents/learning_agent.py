"""
agents/learning_agent.py - 学习资源 Agent

包含 LearningResourceAgent 类，负责：
- TutorialAgent: 基于 DuckDuckGo 搜索生成教程博客
- QueryBot:      多轮 Q&A 对话会话
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

from config import llm_pro
from utils import trim_conversation, save_file


class LearningResourceAgent:
    """处理学习资源相关任务的 Agent。"""

    def __init__(self, prompt):
        """
        Args:
            prompt: ChatPromptTemplate 或消息列表，用作系统提示
        """
        self.model = llm_pro          # 从 config 注入，无需硬编码模型名
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

    def QueryBot(self, user_input: str) -> str:
        """启动多轮 Q&A 对话会话。

        用户输入 'exit' 退出，会话记录保存为 Markdown 文件。

        Args:
            user_input: 初始用户问题

        Returns:
            str: 保存文件的路径
        """
        print("\n开启问答会话。输入 'exit' 结束会话。\n")
        record_QA_session = []
        record_QA_session.append("用户问题: %s \n" % user_input)
        self.prompt.append(HumanMessage(content=user_input))

        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            record_QA_session.append("\n专家回复: %s \n" % response.content)
            self.prompt.append(AIMessage(content=response.content))

            print("*" * 50 + "人工智能助手" + "*" * 50)
            print("\n专家助手回复:", response.content)

            print("*" * 50 + "用户" + "*" * 50)
            user_input = input("\n你的问题: ")
            record_QA_session.append("\n用户问题: %s \n" % user_input)
            self.prompt.append(HumanMessage(content=user_input))

            if user_input.lower() == "exit":
                print("聊天会话已结束。")
                path = save_file("".join(record_QA_session), "Q&A_Doubt_Session")
                print(f"问答记录已保存至: {path}")
                return path
