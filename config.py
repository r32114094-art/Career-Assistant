"""
config.py - LLM 配置模块

使用 DeepSeek API（兼容 OpenAI 格式）替代 Gemini。
所有 Agent 共享此模块导出的 llm / llm_pro 实例。

环境变量（.env）：
    DEEPSEEK_API_KEY  - DeepSeek 平台 API Key
    DEEPSEEK_BASE_URL - 默认 https://api.deepseek.com/v1
    DEEPSEEK_MODEL    - 默认 deepseek-chat（即 DeepSeek-V3）
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

_api_key  = os.getenv("DEEPSEEK_API_KEY", "")
_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
_model    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 主 LLM：用于分类路由（速度快、成本低）────────────────────────
llm = ChatOpenAI(
    model=_model,
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.5,
    verbose=True,
)

# ── 高级 LLM：用于 Tutorial / Resume / JobSearch 生成（质量优先）──
# deepseek-reasoner = DeepSeek-R1（推理增强版），如需更快可改回 deepseek-chat
llm_pro = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_PRO", _model),
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    verbose=True,
)
