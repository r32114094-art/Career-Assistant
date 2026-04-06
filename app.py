"""
app.py - GenAI Career Assistant FastAPI Web 服务（流式输出版）

启动方式：
    conda activate genai-project
    uvicorn app:app --host 0.0.0.0 --port 8000

功能：
    1. WebSocket /ws/{session_id}  — 流式聊天 + HITL 人工审核
    2. 静态文件服务               — 托管 Chat UI 前端页面
    3. 健康检查 /api/health
"""
import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from workflow import build_workflow

# ── 全局线程池 ──────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=8)

# ── 节点中文名映射（用于显示进度） ─────────────────────────
_NODE_LABELS = {
    "categorize":                  "正在理解你的问题...",
    "handle_learning_resource":    "正在匹配学习资源...",
    "handle_interview_preparation":"正在准备面试资料...",
    "handle_resume_making":        "正在生成简历内容...",
    "job_search":                  "正在搜索岗位信息...",
    "job_search_review":           "等待审核搜索结果...",
    "mock_interview":              "正在模拟面试场景...",
    "interview_topics_questions":  "正在整理面试题目...",
    "tutorial_agent":              "正在生成教程内容...",
    "ask_query_bot":               "正在查询答案...",
    "out_of_scope":                "正在处理...",
}

# ── 应用生命周期 ────────────────────────────────────────────
_app_workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_workflow
    print("[START] Building LangGraph workflow ...")
    _app_workflow = build_workflow()
    print("[READY] Workflow ready, waiting for connections ...")
    yield
    print("[STOP] Application shutdown")


app = FastAPI(title="GenAI Career Assistant", lifespan=lifespan)

# ── 静态文件 & 首页 ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "workflow_ready": _app_workflow is not None}


# ── 流式处理核心 ────────────────────────────────────────────

def _stream_graph(workflow, inputs, config, queue: asyncio.Queue, loop):
    """在线程池中运行 LangGraph stream，将事件推入 asyncio Queue。

    使用 stream_mode="messages" 获取逐 token 的 LLM 输出。
    """
    try:
        for event in workflow.stream(inputs, config, stream_mode="updates"):
            # event 是 {node_name: state_update} 的 dict
            for node_name, update in event.items():
                # 发送节点进度
                label = _NODE_LABELS.get(node_name, f"Processing: {node_name}")
                asyncio.run_coroutine_threadsafe(
                    queue.put(("node_progress", {"node": node_name, "label": label})),
                    loop,
                )

                # 从 update 中提取 AI 消息
                messages = update.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.content:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("ai_message", msg.content)),
                            loop,
                        )

        asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
    except Exception as e:
        asyncio.run_coroutine_threadsafe(
            queue.put(("error", str(e))),
            loop,
        )


def _stream_graph_tokens(workflow, inputs, config, queue: asyncio.Queue, loop):
    """在线程池中运行 LangGraph stream，逐 token 推送。

    使用 stream_mode="messages" 获取 LLM 的逐 token 输出。
    """
    try:
        current_node = ""
        for event in workflow.stream(inputs, config, stream_mode="messages"):
            # stream_mode="messages" 返回 (message_chunk, metadata) 元组
            if isinstance(event, tuple) and len(event) == 2:
                msg_chunk, metadata = event
                node_name = metadata.get("langgraph_node", "")

                # 新节点开始 → 发送进度
                if node_name and node_name != current_node:
                    current_node = node_name
                    label = _NODE_LABELS.get(node_name, f"Processing: {node_name}")
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("node_progress", {"node": node_name, "label": label})),
                        loop,
                    )

                # 流式 token
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("token", msg_chunk.content)),
                        loop,
                    )
                elif isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("token", msg_chunk.content)),
                        loop,
                    )

        asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[STREAM-ERROR] {tb}")
        asyncio.run_coroutine_threadsafe(
            queue.put(("error", str(e))),
            loop,
        )


async def _run_streaming(websocket: WebSocket, workflow, inputs, config):
    """运行 LangGraph 流式处理，将结果通过 WebSocket 推送给客户端。

    优先尝试 token 级流式（stream_mode="messages"），
    如失败则回退到节点级流式（stream_mode="updates"）。
    """
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    # 通知前端开始流式
    await websocket.send_json({"type": "stream_start"})

    # 尝试 token 级流式
    future = loop.run_in_executor(
        _executor, _stream_graph_tokens, workflow, inputs, config, queue, loop
    )

    collected_content = ""
    has_tokens = False

    try:
        while True:
            try:
                msg_type, content = await asyncio.wait_for(queue.get(), timeout=180)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Processing timeout (3 minutes). Please try again.",
                })
                break

            if msg_type == "token":
                has_tokens = True
                collected_content += content
                await websocket.send_json({"type": "ai_token", "token": content})

            elif msg_type == "ai_message":
                # 整块消息（来自 updates 模式的 fallback）
                collected_content = content
                await websocket.send_json({"type": "ai_message", "content": content})

            elif msg_type == "node_progress":
                await websocket.send_json({
                    "type": "node_progress",
                    "node": content["node"],
                    "label": content["label"],
                })

            elif msg_type == "done":
                break

            elif msg_type == "error":
                # Token 流式失败，回退到节点级
                if not has_tokens:
                    print("[FALLBACK] Token streaming failed, falling back to updates mode")
                    queue2 = asyncio.Queue()
                    future2 = loop.run_in_executor(
                        _executor, _stream_graph, workflow, inputs, config, queue2, loop
                    )
                    while True:
                        msg_type2, content2 = await queue2.get()
                        if msg_type2 == "ai_message":
                            collected_content = content2
                            await websocket.send_json({
                                "type": "ai_message",
                                "content": content2,
                            })
                        elif msg_type2 == "node_progress":
                            await websocket.send_json({
                                "type": "node_progress",
                                "node": content2["node"],
                                "label": content2["label"],
                            })
                        elif msg_type2 == "done":
                            break
                        elif msg_type2 == "error":
                            await websocket.send_json({
                                "type": "error",
                                "content": f"Error: {content2}",
                            })
                            break
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error: {content}",
                    })
                break

    finally:
        await websocket.send_json({"type": "stream_end"})

    return collected_content


def _sync_get_state(workflow, config):
    return workflow.get_state(config)


# ── WebSocket 聊天端点 ──────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """WebSocket 聊天处理 — 流式输出 + HITL 审核

    消息协议（JSON）：
        客户端 → 服务端：
            {"type": "message",       "content": "用户消息"}
            {"type": "hitl_response", "decision": "approve|reject|自定义文本"}

        服务端 → 客户端：
            {"type": "stream_start"}
            {"type": "ai_token",      "token": "单个token文本"}
            {"type": "ai_message",    "content": "完整AI消息"}
            {"type": "stream_end"}
            {"type": "node_progress", "node": "节点名", "label": "中文进度"}
            {"type": "hitl_request",  "instruction": "提示", "preview": "预览"}
            {"type": "thinking",      "status": true/false}
            {"type": "error",         "content": "错误信息"}
            {"type": "connected",     "session_id": "xxx"}
    """
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    loop = asyncio.get_event_loop()

    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
    })

    # ── 恢复历史对话记录 ──────────────────────────────
    try:
        state_snapshot = await loop.run_in_executor(
            _executor, _sync_get_state, _app_workflow, config
        )
        if state_snapshot and state_snapshot.values:
            history_messages = state_snapshot.values.get("messages", [])
            history = []
            for msg in history_messages:
                if isinstance(msg, HumanMessage) and msg.content:
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    history.append({"role": "ai", "content": msg.content})
            if history:
                await websocket.send_json({
                    "type": "chat_history",
                    "messages": history,
                })
    except Exception as e:
        print(f"[WARN] Failed to load history for {session_id}: {e}")

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "message":
                user_text = data.get("content", "").strip()
                if not user_text:
                    continue

                try:
                    # 流式处理
                    await _run_streaming(
                        websocket, _app_workflow,
                        {"messages": [("user", user_text)]},
                        config,
                    )

                    # 检查 HITL 中断
                    state_snapshot = await loop.run_in_executor(
                        _executor, _sync_get_state, _app_workflow, config
                    )
                    if state_snapshot.next:
                        payload = None
                        for task in state_snapshot.tasks:
                            if task.interrupts:
                                payload = task.interrupts[0].value
                                break
                        if payload:
                            await websocket.send_json({
                                "type": "hitl_request",
                                "instruction": payload.get("instruction",
                                                           "Need your confirmation"),
                                "preview": payload.get("preview", ""),
                            })

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error: {str(e)}",
                    })

            elif msg_type == "hitl_response":
                decision = data.get("decision", "").strip()
                if decision.lower() in ("y", "yes", "approve", ""):
                    resume_value = "approve"
                elif decision.lower() in ("n", "no", "reject"):
                    resume_value = "reject"
                else:
                    resume_value = decision

                try:
                    await _run_streaming(
                        websocket, _app_workflow,
                        Command(resume=resume_value),
                        config,
                    )

                    # 再次检查 HITL
                    state_snapshot = await loop.run_in_executor(
                        _executor, _sync_get_state, _app_workflow, config
                    )
                    if state_snapshot.next:
                        payload = None
                        for task in state_snapshot.tasks:
                            if task.interrupts:
                                payload = task.interrupts[0].value
                                break
                        if payload:
                            await websocket.send_json({
                                "type": "hitl_request",
                                "instruction": payload.get("instruction",
                                                           "Need your confirmation"),
                                "preview": payload.get("preview", ""),
                            })

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Resume error: {str(e)}",
                    })

    except WebSocketDisconnect:
        print(f"[DISCONNECT] session={session_id}")
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] WebSocket error session={session_id}: {e}")


# ── 本地启动入口 ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
