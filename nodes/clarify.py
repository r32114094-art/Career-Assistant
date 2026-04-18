"""
nodes/clarify.py - 澄清节点

当路由置信度低或 needs_clarification=True 时触发，向用户发出澄清问题。
澄清后对话自然进入下一轮，categorize 节点重新分类。
clarify_count 记录澄清轮次，上限 2 次。
"""
from langchain_core.messages import AIMessage
from state import State


def clarify(state: State) -> dict:
    """向用户发送澄清问题，并递增 clarify_count。"""
    rd = state.get("routing_decision") or {}
    question = rd.get("clarify_question") or "您能描述一下您想要什么帮助吗？"
    print(f"澄清轮次 {state.get('clarify_count', 0) + 1}，发送澄清问题")
    return {
        "messages": [AIMessage(content=question)],
        "clarify_count": state.get("clarify_count", 0) + 1,
    }
