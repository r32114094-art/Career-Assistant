"""
state.py - 状态定义模块

使用 TypedDict 定义 LangGraph 工作流中共享的状态结构。
"""
from typing import TypedDict


class State(TypedDict):
    """LangGraph 工作流的状态结构

    Attributes:
        query:    用户原始查询字符串
        category: LLM 分类后的类别标签
        response: 最终响应的文件路径或内容
    """
    query: str
    category: str
    response: str
