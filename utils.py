"""
utils.py - 工具函数模块

包含：
- trim_conversation: 裁剪对话历史，保留最近 10 条消息
- save_file:         将内容保存为带时间戳的 Markdown 文件
- show_md_file:      读取并打印 Markdown 文件内容（控制台版本）
"""
import os
from datetime import datetime
from langchain_core.messages import trim_messages


def trim_conversation(prompt):
    """裁剪对话历史，保留最近 10 条消息。"""
    max_messages = 10
    return trim_messages(
        prompt,
        max_tokens=max_messages,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


def save_file(data: str, filename: str) -> str:
    """将数据保存到带时间戳的 Markdown 文件。

    Args:
        data:     要保存的字符串内容
        filename: 文件名前缀（不含扩展名）

    Returns:
        str: 保存文件的完整路径
    """
    folder_name = "Agent_output"
    os.makedirs(folder_name, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename}_{timestamp}.md"
    file_path = os.path.join(folder_name, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)
        print(f"文件 '{file_path}' 创建成功。")

    return file_path


def show_md_file(file_path: str) -> None:
    """读取并打印 Markdown 文件内容到控制台。

    在 Jupyter 环境中可替换为 display(Markdown(content))。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("\n" + "=" * 60)
    print(content)
    print("=" * 60 + "\n")


def get_current_time() -> str:
    """动态获取当前物理时间用于注入系统提示词"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
