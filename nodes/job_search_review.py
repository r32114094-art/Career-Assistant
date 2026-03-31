"""
nodes/job_search_review.py - 求职结果 HITL 审核节点

包含一个 LangGraph 节点函数：
- job_search_review: 读取 state["pending_job_results"]，通过 interrupt() 挂起图，
                     由人类审核后决定保存、拒绝或追加修改意见。

【为何单独成节点】
  interrupt() 触发后，LangGraph 在 resume 时会从本节点头部重新执行。
  本节点只做"读 state → interrupt → 分支"三步，所有操作均无副作用（文件写入在
  approve 分支，且只在用户确认后才执行），因此 resume 重放完全安全。
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from state import State
from utils import save_file


def job_search_review(state: State) -> State:
    """HITL 人工审核节点：挂起图，等待用户对搜索结果的确认。

    决策逻辑：
        "approve" → 写入文件，返回成功消息
        "reject"  → 取消，返回取消消息
        其他文本  → 把用户的修改意见作为新 HumanMessage 追加，触发新一轮路由
    """
    pending = state.get("pending_job_results", "")

    # 没有待审核结果（对话轮次不涉及完整搜索）直接跳过
    if not pending:
        return {}

    # 动态中断：只有此节点包含 interrupt()，resume 重放安全
    human_decision = interrupt({
        "instruction": "岗位已收集，是否满意？(输入 'y'/回车 确认保存，否则填写过滤要求如'只要远程的'):",
        "preview": pending[:600],
    })

    if human_decision == "approve":
        path = save_file(pending, "Job_search")
        return {
            "messages": [AIMessage(content=f"最终版岗位精选列表已为您保存至: {path}\n（搜索过滤完毕，流程结束）")],
            "pending_job_results": "",
            "response": path,
        }
    elif human_decision == "reject":
        return {
            "messages": [AIMessage(content="已取消保存。如需重新搜索，请重新描述您的需求。")],
            "pending_job_results": "",
            "response": "已取消。",
        }
    else:
        # 用户给出了修改意见，追加到消息历史，下一轮图会重新路由到 job_search
        return {
            "messages": [HumanMessage(content=human_decision)],
            "pending_job_results": "",
            "response": human_decision,
        }
