"""
agents/interview_agent.py - 面试 Agent

包含 InterviewAgent 类，负责：
- Interview_questions: 面试题目问答会话
- Mock_Interview:      模拟面试会话
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

from config import llm          # 面试场景用标准模型即可
from utils import trim_conversation, save_file


class InterviewAgent:
    """处理面试准备相关任务的 Agent。"""

    def __init__(self, prompt):
        """
        Args:
            prompt: ChatPromptTemplate 或消息列表，用作系统提示
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

    def Interview_questions(self, user_input: str) -> str:
        """通过对话循环提供面试题目，并保存为 Markdown 文件。

        Args:
            user_input: 用户初始问题

        Returns:
            str: 保存文件的路径
        """
        chat_history = []
        questions_bank = ""
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        while True:
            print("\n开启面试题准备。输入 'exit' 结束会话。\n")
            if user_input.lower() == "exit":
                print("对话结束。再见！")
                break

            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            questions_bank += (
                str(response.get("output")).replace("```markdown", "").strip() + "\n"
            )

            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            user_input = input("你: ")

        path = save_file(questions_bank, "Interview_questions")
        print(f"面试题已保存至: {path}")
        return path

    def Mock_Interview(self) -> str:
        """启动模拟面试会话，并保存记录为 Markdown 文件。

        Returns:
            str: 保存文件的路径
        """
        print("\n开启模拟面试。输入 'exit' 结束会话。\n")

        initial_message = "我准备好面试了。\n"
        interview_record = []
        interview_record.append("候选人: %s \n" % initial_message)
        self.prompt.append(HumanMessage(content=initial_message))

        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            self.prompt.append(AIMessage(content=response.content))

            print("\n面试官:", response.content)
            interview_record.append("\n面试官: %s \n" % response.content)

            user_input = input("\n候选人: ")
            interview_record.append("\n候选人: %s \n" % user_input)
            self.prompt.append(HumanMessage(content=user_input))

            if user_input.lower() == "exit":
                print("面试会话结束。")
                path = save_file("".join(interview_record), "Mock_Interview")
                print(f"模拟面试记录已保存至: {path}")
                return path
