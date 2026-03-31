"""
main.py - GenAI Career Assistant 主入口

运行方式：
    python main.py

功能：
    1. 构建 LangGraph 工作流（含 MemorySaver 持久化）
    2. 通过 thread_id 维持跨轮次的对话记忆
    3. 每次用户输入触发一次完整的 图流转 (START → ... → END)
    4. 提取最后一条 AIMessage 作为输出展示
"""
from langchain_core.messages import AIMessage
from langgraph.types import Command

from workflow import build_workflow


def main():
    print("=" * 60)
    print("  GenAI 职业助手 Agent ")
    print("  你开启生成式 AI 职业生涯的终极指南！ 🚀")
    print("=" * 60)
    print()
    print("我可以在以下方面帮助你：")
    print("  1. 📚 学习生成式 AI (教程与问答)")
    print("  2. 📄 简历制作与评审")
    print("  3. 🎯 面试准备 (题目与模拟面试)")
    print("  4. 🔍 求职辅助")
    print()

    # 构建工作流（已挂载 MemorySaver）
    app = build_workflow()

    # 通过 thread_id 维持会话级记忆
    # 同一个 thread_id 的多次 invoke 会自动累积 messages
    config = {"configurable": {"thread_id": "session_001"}}

    while True:
        print("-" * 60)
        query = input("你的问题 (输入 'quit' 退出): ").strip()
        if query.lower() in ("quit", "q"):
            print("再见！祝你在 GenAI 的职业旅程中好运！ 🎉")
            break
        if not query:
            print("请输入有效的问题。")
            continue

        # 每次 invoke 都携带 config，Checkpointer 自动恢复/保存状态
        results = app.invoke(
            {"messages": [("user", query)]},
            config,
        )

        # ── HITL 审核循环（兼容 langgraph 0.2.x）──────────────────────
        # 0.2.x 中 interrupt() 不在 results 里放 __interrupt__ key，
        # 而用 app.get_state().next 判断图是否被挂起，
        # 中断 payload 从 state_snapshot.tasks[].interrupts 中读取
        state_snapshot = app.get_state(config)
        while state_snapshot.next:
            payload = None
            for task in state_snapshot.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    break

            if payload is None:
                break

            print(f"\n💡 助手: {payload.get('instruction', '需要您的确认')}")
            if payload.get("preview"):
                print(payload["preview"])
            print("-" * 60)

            decision = input("你的决定 (输入 'y'/回车 确认保存，'n' 拒绝，或直接输入修改要求): ").strip()
            if decision.lower() in ("y", "yes", ""):
                resume_value = "approve"
            elif decision.lower() in ("n", "no", "quit"):
                resume_value = "reject"
            else:
                resume_value = decision

            # 将用户的决定传递回中断节点，重新推进图执行
            results = app.invoke(Command(resume=resume_value), config)
            state_snapshot = app.get_state(config)

        # 从结果中提取最后一条 AI 回复
        ai_response = None
        for msg in reversed(results.get("messages", [])):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break

        if ai_response:
            print(f"\n🤖 助手: {ai_response}")
        else:
            print("\n⚠️ 未获取到有效回复，请重试。")


if __name__ == "__main__":
    main()
