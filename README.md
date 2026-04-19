# GenAI 职业助手

基于 **LangChain + LangGraph** 构建的多轮对话 AI 职业发展助手，能够将用户问题路由至五个领域并跨会话保持持久记忆。

## 功能特性

- **智能路由** — 将每条消息分类至五个领域（学习、简历、面试、求职、超出范围）
- **学习资源** — 教程生成与知识问答
- **简历制作** — 全流程简历起草与修改
- **面试准备** — 模拟面试与题库查询
- **职位搜索** — 通过 SerpAPI 实时搜索，结果经 Human-in-the-Loop（HITL）审核后下发
- **多轮记忆** — 对话历史按 `thread_id` 持久化（本地 SQLite，可选 PostgreSQL）
- **双端口** — CLI 命令行（`main.py`）与 WebSocket Web 服务器（`app.py`）

## 系统架构

```
START → categorize → route_query（条件分支）
    ├─ 1: handle_learning_resource → [tutorial | qa]
    ├─ 2: handle_resume_making    → ResumeMaker Agent
    ├─ 3: handle_interview        → [mock_interview | question_bank]
    ├─ 4: job_search              → JobSearch Agent → job_search_review（HITL 中断）
    └─ 5: out_of_scope
→ END
```

核心为 [workflow.py](workflow.py) 中的 `StateGraph`，状态结构定义于 [state.py](state.py)。  
两个 LLM 实例配置于 [config.py](config.py)：
- `llm`（`deepseek-chat`）— 快速模型，用于路由与分类
- `llm_pro`（`deepseek-reasoner`）— 强力模型，用于内容生成

### 目录结构

```
├── agents/          # 业务逻辑：LearningResourceAgent、ResumeMaker、InterviewAgent、JobSearch
├── nodes/           # LangGraph 节点函数，将 Agent 接入 StateGraph
├── eval/            # 路由准确率评测（数据集 + 脚本 + 结果）
├── finetune/        # SFT→DPO→GRPO 训练流程，训练本地 Qwen2.5-3B 路由模型（进行中）
│   ├── configs/     # 训练配置（sft/dpo/grpo）
│   ├── data/        # 生成的训练数据
│   ├── scripts/     # 按编号排列的流程脚本（01_–06_）
│   ├── eval/        # 微调模型评测
│   ├── inference/   # 可直接替换的路由推理模块（fc_router.py）
│   └── docs/        # 训练流程文档
├── docs/            # 设计文档与优化计划
├── static/          # Web 模式前端（HTML/CSS/JS）
├── app.py           # FastAPI + WebSocket 服务器
├── main.py          # CLI 入口
├── workflow.py      # LangGraph StateGraph 定义
├── router.py        # 条件边 / 路由逻辑
├── state.py         # State TypedDict
├── config.py        # LLM 实例与环境配置
└── utils.py         # 公共工具函数
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

必填配置项：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key — 在 [platform.deepseek.com](https://platform.deepseek.com) 获取 |
| `SERPAPI_API_KEY` | SerpAPI Key，用于职位搜索 — [serpapi.com](https://serpapi.com) |
| `LANGSMITH_API_KEY` | 可选 — LangSmith 链路追踪，在 [smith.langchain.com](https://smith.langchain.com) 获取 |
| `DATABASE_URL` | 可选 — PostgreSQL 连接 URL，用于云端记忆存储（不填则使用本地 SQLite） |

### 3. 启动

**命令行模式：**
```bash
python main.py
```

**Web 服务器模式：**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
启动后在浏览器访问 `http://localhost:8000`。

## 评测

```bash
# 路由准确率评测（v1 基线 vs v2 结构化路由）
python eval/eval_routing.py
```

评测结果和混淆矩阵保存至 `eval/results/`。

## 微调流程（进行中）

`finetune/` 目录包含完整的 **SFT → DPO → GRPO** 训练流程，目标是训练本地 [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) 模型，作为现有 LLM 路由器的可替换方案。

> **当前状态：** 数据生成与训练脚本已完成，模型训练与集成待进行。

按编号顺序执行脚本：

```bash
python finetune/scripts/01_generate_sft_data.py
python finetune/scripts/01b_split_dataset.py
python finetune/scripts/02_train_sft.py
python finetune/scripts/03_generate_dpo_data.py
python finetune/scripts/04_train_dpo.py
python finetune/scripts/05_train_grpo.py
python finetune/scripts/06_merge_and_export.py
```

详细说明见 [finetune/docs/training_pipeline.md](finetune/docs/training_pipeline.md)。

训练完成后，可通过 [finetune/inference/fc_router.py](finetune/inference/fc_router.py) 无缝替换路由模块，无需修改主图代码。

## Human-in-the-Loop（HITL）

职位搜索结果在下发前会暂停等待人工审核。CLI 模式下在 `main.py` 中交互处理；Web 模式下服务器发送 `hitl_request` WebSocket 事件，前端响应 `approve`、`reject` 或 `modify`。

## 记忆存储

默认将对话历史存储在本地 SQLite 文件 `data/memory.db`（自动创建，已加入 .gitignore）。如需使用 PostgreSQL，在 `.env` 中设置 `DATABASE_URL`。

## 技术栈

- [LangChain](https://python.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/)
- [DeepSeek API](https://platform.deepseek.com)（V3 用于路由，R1 用于生成）
- [SerpAPI](https://serpapi.com)（职位搜索）
- [FastAPI](https://fastapi.tiangolo.com/) + WebSocket（Web 模式）
- SQLite / PostgreSQL（记忆持久化）
