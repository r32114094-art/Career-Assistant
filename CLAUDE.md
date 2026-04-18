# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenAI Career Assistant is a multi-turn conversational AI agent for career development, built with LangChain + LangGraph. It routes user queries across five domains: learning resources, resume making, interview preparation, job search, and out-of-scope.

## Setup & Running

```bash
pip install -r requirements.txt
```

Required `.env` variables:
```
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat          # Used for routing/classification
DEEPSEEK_MODEL_PRO=deepseek-reasoner  # Used for generation tasks
SERPAPI_API_KEY=
DATABASE_URL=postgresql://...         # Supabase PostgreSQL for session persistence
LANGSMITH_TRACING=true                # Optional
LANGSMITH_API_KEY=                    # Optional
```

**CLI mode:** `python main.py`
**Web server:** `uvicorn app:app --host 0.0.0.0 --port 8000`

## Testing & Evaluation

```bash
# Routing accuracy evaluation
python eval/eval_routing.py

# Fine-tuned model evaluation (multi-metric)
python finetune/eval/evaluate.py
```

The routing test dataset is at [eval/routing_dataset.json](eval/routing_dataset.json).

## Architecture

### LangGraph State Machine

The core is a `StateGraph` in [workflow.py](workflow.py) with a `State` TypedDict ([state.py](state.py)) holding `messages`, `category`, `response`, and `pending_job_results`. Multi-turn memory is handled automatically by `PostgresSaver` (checkpointer) keyed by `thread_id`.

**Graph flow:**
```
START → categorize → route_query (conditional)
    ├─ 1: handle_learning_resource → [tutorial | qa]
    ├─ 2: handle_resume_making → ResumeMaker Agent
    ├─ 3: handle_interview_preparation → [mock_interview | question_bank]
    ├─ 4: job_search → JobSearch Agent → job_search_review (HITL interrupt)
    └─ 5: out_of_scope
→ END
```

### Two LLM Instances

Defined in [config.py](config.py):
- `llm` (`deepseek-chat`) — fast, used for routing/classification nodes
- `llm_pro` (`deepseek-reasoner`) — powerful, used by agents for content generation

### Nodes vs Agents

- **[nodes/](nodes/)** — LangGraph node functions that integrate into the StateGraph; each receives `State` and returns a partial state update
- **[agents/](agents/)** — Business logic classes (`LearningResourceAgent`, `ResumeMaker`, `InterviewAgent`, `JobSearch`) that use `AgentExecutor` + SerpAPI tools or direct LLM calls; nodes instantiate and call these

### HITL (Human-in-the-Loop)

The `job_search_review` node in [nodes/job_search_review.py](nodes/job_search_review.py) uses LangGraph's `interrupt()` to pause execution. Resumed via `Command(resume={"action": "approve"|"reject"|"modify"})`. The CLI in [main.py](main.py) handles this manually; the web server in [app.py](app.py) emits a `hitl_request` WebSocket event and waits for client response.

### Streaming (Web Mode)

[app.py](app.py) runs LangGraph in a `ThreadPoolExecutor`, yielding WebSocket events: `node_progress`, `ai_token`, `hitl_request`, `complete`, `error`. Tries `stream_mode="messages"` (token-level) and falls back to `stream_mode="updates"` (node-level).

### Routing System

[router.py](router.py) defines conditional edge functions. The current implementation uses LLM + regex to set `state["category"]` (1–5). A structured routing refactor is planned (see [docs/P0-1_结构化路由改进实施计划.md](docs/P0-1_结构化路由改进实施计划.md)).

### Fine-tuning Pipeline

[finetune/](finetune/) contains a complete SFT→DPO→GRPO pipeline to train a Qwen2.5-3B replacement for the LLM-based router. Scripts run in numbered order (`01_`–`06_`). The trained router is drop-in replaceable via [finetune/inference/fc_router.py](finetune/inference/fc_router.py). See [finetune/docs/training_pipeline.md](finetune/docs/training_pipeline.md) for the full pipeline guide including documented bug fixes.

## Key Utilities

- [utils.py](utils.py): `trim_conversation()` keeps last 10 messages; `save_file()` persists job results; `get_current_time()` injected into all agent prompts
- [state.py](state.py): `get_latest_user_text()`, `get_chat_history()` for extracting context from `messages`
