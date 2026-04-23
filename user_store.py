"""
user_store.py - 轻量用户存储（文件系统 JSON）

小项目级用户态存储方案：无需鉴权、无需业务 DB。
- 每个用户一个 JSON 文件：data/users/{user_id}.json
- 存储用户画像（跨会话持久化）+ 会话元信息列表
- thread_id 命名规则：{user_id}::{session_id}，复用 LangGraph Checkpointer

数据结构：
{
    "user_id": "barry_2025",
    "profile": {...UserProfile...},
    "sessions": [
        {
            "session_id": "abc123",
            "title": "关于 RAG 的学习",
            "created_at": "2026-04-21T10:00:00",
            "updated_at": "2026-04-21T10:30:00"
        }
    ]
}
"""
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Optional

_USERS_DIR = os.path.join("data", "users")
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_user_id(user_id: str) -> str:
    """过滤 user_id 防止路径穿越，只保留字母数字下划线中划线。"""
    return "".join(c for c in user_id if c.isalnum() or c in ("_", "-")).strip() or "anonymous"


def _user_file(user_id: str) -> str:
    os.makedirs(_USERS_DIR, exist_ok=True)
    return os.path.join(_USERS_DIR, f"{_safe_user_id(user_id)}.json")


def _load(user_id: str) -> dict:
    path = _user_file(user_id)
    if not os.path.exists(path):
        return {"user_id": _safe_user_id(user_id), "profile": {}, "sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"user_id": _safe_user_id(user_id), "profile": {}, "sessions": []}


def _save(user_id: str, data: dict) -> None:
    path = _user_file(user_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ── 会话管理 ─────────────────────────────────────────────────

def list_sessions(user_id: str) -> list:
    """返回按更新时间倒序排列的会话列表。"""
    with _LOCK:
        data = _load(user_id)
    sessions = data.get("sessions", [])
    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def create_session(user_id: str, title: str = "新对话") -> dict:
    """为用户创建新会话，返回完整的会话元信息。"""
    with _LOCK:
        data = _load(user_id)
        session = {
            "session_id": uuid.uuid4().hex[:12],
            "title": title,
            "created_at": _now(),
            "updated_at": _now(),
        }
        data.setdefault("sessions", []).append(session)
        _save(user_id, data)
    return session


def delete_session(user_id: str, session_id: str) -> bool:
    """删除会话元信息（不触及 LangGraph checkpoint 数据）。"""
    with _LOCK:
        data = _load(user_id)
        sessions = data.get("sessions", [])
        new_sessions = [s for s in sessions if s.get("session_id") != session_id]
        if len(new_sessions) == len(sessions):
            return False
        data["sessions"] = new_sessions
        _save(user_id, data)
    return True


def update_session_meta(user_id: str, session_id: str, **fields) -> None:
    """更新会话元信息（title / updated_at 等）。"""
    with _LOCK:
        data = _load(user_id)
        for s in data.get("sessions", []):
            if s.get("session_id") == session_id:
                s.update(fields)
                s["updated_at"] = _now()
                break
        _save(user_id, data)


def touch_session(user_id: str, session_id: str) -> None:
    """轻量更新会话最后活跃时间。若会话不存在则自动补建（兼容旧链接）。"""
    with _LOCK:
        data = _load(user_id)
        sessions = data.setdefault("sessions", [])
        for s in sessions:
            if s.get("session_id") == session_id:
                s["updated_at"] = _now()
                _save(user_id, data)
                return
        sessions.append({
            "session_id": session_id,
            "title": "新对话",
            "created_at": _now(),
            "updated_at": _now(),
        })
        _save(user_id, data)


# ── 画像管理 ─────────────────────────────────────────────────

def get_profile(user_id: str) -> dict:
    with _LOCK:
        return _load(user_id).get("profile", {}) or {}


def save_profile(user_id: str, profile: dict) -> None:
    """覆盖写入用户画像（调用方负责合并）。"""
    with _LOCK:
        data = _load(user_id)
        data["profile"] = profile or {}
        _save(user_id, data)


# ── thread_id 辅助 ───────────────────────────────────────────

THREAD_SEP = "::"


def make_thread_id(user_id: str, session_id: str) -> str:
    return f"{_safe_user_id(user_id)}{THREAD_SEP}{session_id}"


def parse_thread_id(thread_id: str) -> Optional[tuple]:
    """从 thread_id 解析出 (user_id, session_id)，失败返回 None。"""
    if THREAD_SEP not in thread_id:
        return None
    user_id, session_id = thread_id.split(THREAD_SEP, 1)
    return user_id, session_id
