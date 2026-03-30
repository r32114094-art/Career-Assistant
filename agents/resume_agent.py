"""
agents/resume_agent.py - 简历制作 Agent

包含 ResumeMaker 类，通过多轮对话收集用户信息并生成专业简历。
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage

from config import llm_pro
from utils import save_file


class ResumeMaker:
    """通过对话循环创建专业简历的 Agent。"""

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

    def Create_Resume(self, user_input: str) -> str:
        """通过多轮对话收集用户信息并生成简历 Markdown 文件。

        Args:
            user_input: 用户初始请求

        Returns:
            str: 保存文件的路径
        """
        chat_history = []
        response = None

        while True:
            print("\n开启简历制作会话。输入 'exit' 结束会话。\n")
            if user_input.lower() == "exit":
                print("对话结束。再见！")
                break

            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            chat_history.extend([HumanMessage(content=user_input), response["output"]])

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            user_input = input("你: ")

        if response is None:
            return ""

        content = str(response.get("output")).replace("```markdown", "").strip()
        path = save_file(content, "Resume")
        print(f"简历已保存至: {path}")
        return path
