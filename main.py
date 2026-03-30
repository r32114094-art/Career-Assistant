"""
main.py - GenAI Career Assistant 主入口

运行方式：
    python main.py

功能：
    1. 构建 LangGraph 工作流
    2. 提示用户输入查询
    3. 通过工作流处理查询并返回结果
"""
from typing import Dict

from workflow import build_workflow


def run_user_query(app, query: str) -> Dict[str, str]:
    """通过 LangGraph 工作流处理用户查询。

    Args:
        app:   编译后的 LangGraph 工作流应用
        query: 用户查询字符串

    Returns:
        Dict[str, str]: 包含 category 和 response 的字典
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "response": results["response"],
    }


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

    # 构建工作流
    app = build_workflow()

    while True:
        print("-" * 60)
        query = input("你的问题 (输入 'quit' 退出): ").strip()
        if query.lower() in ("quit", "q"):
            print("再见！祝你在 GenAI 的职业旅程中好运！ 🎉")
            break
        if not query:
            print("请输入有效的问题。")
            continue

        result = run_user_query(app, query)
        print(f"\n✅ 类别: {result['category']}")
        print(f"✅ 输出已保存至: {result['response']}")


if __name__ == "__main__":
    main()
