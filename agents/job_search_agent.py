"""
agents/job_search_agent.py - 求职搜索 Agent

包含 JobSearch 类，利用 DuckDuckGo 搜索市场职位并整理为 Markdown。
"""
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from config import llm_pro
from utils import save_file


class JobSearch:
    """通过搜索工具查找并整理 GenAI 相关职位的 Agent。"""

    def __init__(self, prompt):
        """
        Args:
            prompt: ChatPromptTemplate，仅需 {result} 插槽
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

    def find_jobs(self, user_input: str) -> str:
        """搜索职位并整理为 Markdown 文件。

        Args:
            user_input: 用户查询（含地点、职位等关键词）

        Returns:
            str: 保存文件的路径
        """
        chat_history = []
        current_input = user_input
        
        print("\n开启求职要求确认... 输入 'exit' 结束会话。\n")
        while True:
            response = self.agent_executor.invoke({
                "input": current_input,
                "chat_history": chat_history
            })
            
            ai_msg = response["output"]
            # To make the console clean since we have verbose=True, we rely on the final output.
            
            if "[TASK_DONE]" in ai_msg:
                content = ai_msg.replace("[TASK_DONE]", "").replace("```markdown", "").replace("```", "").strip()
                path = save_file(content, "Job_search")
                print(f"职位信息已保存至: {path}")
                return path
            
            # 走到这里说明还没收集满信息，记录历史并向用户发问
            chat_history.extend([
                HumanMessage(content=current_input),
                AIMessage(content=ai_msg)
            ])
            
            current_input = input("你: ")
            if current_input.lower() == 'exit':
                return "已取消搜索。"
