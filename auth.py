"""
auth.py - 轻量认证模块（统一口令 + 内存 Token）

模式：用户自定义用户名 + 服务端统一口令（邀请码）
口令来源：data/credentials.json 中的 "access_codes" 列表

使用方式：
    from auth import login, verify_token, logout

    token = login("barry", "genai2026")   # 用户名自取，口令需匹配列表中任一
    user_id = verify_token(token)          # 合法返回 user_id，否则 None
    logout(token)                          # 销毁 token
"""
import json
import os
import re
import secrets
from typing import Optional

# ── 口令列表 ──────────────────────────────────────────────────
_ACCESS_CODES: set[str] = set()

# ── 活跃 Token 映射（内存，重启后失效） ───────────────────────
_TOKENS: dict[str, str] = {}  # {token: user_id}


def _load_access_codes() -> None:
    """加载口令列表。优先级：环境变量 ACCESS_CODES > data/credentials.json"""
    global _ACCESS_CODES

    # 1. 优先从环境变量读取（部署环境用，逗号分隔）
    env_codes = os.getenv("ACCESS_CODES", "").strip()
    if env_codes:
        _ACCESS_CODES = set(c.strip() for c in env_codes.split(",") if c.strip())
        print(f"[Auth] 从环境变量加载 {len(_ACCESS_CODES)} 个访问口令")
        return

    # 2. 回退到 JSON 文件（本地开发用）
    path = os.path.join("data", "credentials.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        codes = config.get("access_codes", [])
        if not codes and config.get("access_code"):
            codes = [config["access_code"]]
        _ACCESS_CODES = set(codes)
        print(f"[Auth] 从文件加载 {len(_ACCESS_CODES)} 个访问口令")
    else:
        print("[Auth] 未找到口令配置")


# 启动时加载
_load_access_codes()


def _sanitize_username(username: str) -> str:
    """清洗用户名：只保留字母数字下划线中划线，2~30 字符。"""
    cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '', username.strip().lower())
    return cleaned


def login(username: str, password: str) -> Optional[str]:
    """校验用户名和口令。口令匹配列表中任一即可登录。"""
    username = _sanitize_username(username)
    if not username or len(username) < 2:
        return None
    if not _ACCESS_CODES or password not in _ACCESS_CODES:
        return None

    # 生成随机 token
    token = secrets.token_hex(24)
    _TOKENS[token] = username
    return token


def verify_token(token: str) -> Optional[str]:
    """验证 token。合法返回 user_id（即 username），否则返回 None。"""
    if not token:
        return None
    return _TOKENS.get(token)


def guest_login() -> tuple[str, str]:
    """游客登录：自动生成随机用户名，无需口令。返回 (token, user_id)。"""
    user_id = f"guest_{secrets.token_hex(4)}"
    token = secrets.token_hex(24)
    _TOKENS[token] = user_id
    return token, user_id


def logout(token: str) -> bool:
    """销毁 token。返回是否成功。"""
    return _TOKENS.pop(token, None) is not None
