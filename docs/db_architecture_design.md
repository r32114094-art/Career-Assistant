# GenAI Career Assistant — PostgreSQL 数据库架构重构方案

> 面试回答场景：面试官问"如果重新架构数据库，你会怎么设计？"

## 一、现状分析

当前数据库只有 LangGraph `PostgresSaver` 自动创建的 3 张框架表（`checkpoints` / `checkpoint_blobs` / `checkpoint_writes`），**没有任何自建业务表**。所有数据（对话历史、用户画像、路由决策）都被序列化塞进 checkpoint 的 `channel_values` JSON blob 里。

**核心问题**：
- 无法独立查询用户画像、对话统计等业务数据
- 无法做跨会话的用户数据聚合（如"用户在多个会话中最常练习的面试方向"）
- 无法支持多用户认证和会话管理
- 数据分析和运营完全无从下手

---

## 二、重构后的表结构设计

```mermaid
erDiagram
    users ||--o{ sessions : "1:N"
    sessions ||--o{ messages : "1:N"
    users ||--|| user_profiles : "1:1"
    users ||--o{ interview_records : "1:N"
    users ||--o{ job_search_logs : "1:N"
    users ||--o{ resume_snapshots : "1:N"

    users {
        uuid id PK
        varchar email UK
        varchar display_name
        varchar auth_provider
        timestamptz created_at
        timestamptz last_active_at
    }

    sessions {
        uuid id PK
        uuid user_id FK
        varchar thread_id UK "LangGraph thread_id"
        varchar title "会话标题（LLM自动摘要）"
        varchar session_type "general | interview | job_search"
        boolean is_active
        timestamptz created_at
        timestamptz updated_at
    }

    messages {
        bigint id PK
        uuid session_id FK
        varchar role "user | ai | system | tool"
        text content
        jsonb metadata "路由决策、token用量等"
        timestamptz created_at
    }

    user_profiles {
        uuid user_id PK_FK
        varchar target_role
        varchar skill_level
        integer years_of_experience
        text background
        text[] skills
        text[] interests
        varchar preferred_location
        varchar preferred_work_type
        jsonb extra "可扩展字段"
        timestamptz updated_at
    }

    interview_records {
        bigint id PK
        uuid user_id FK
        uuid session_id FK
        varchar interview_type "mock | topic_qa"
        varchar topic "e.g. 系统设计, RAG, LLM"
        integer score "LLM评分 0-100"
        text feedback_summary
        jsonb qa_pairs "问答对原文"
        timestamptz created_at
    }

    job_search_logs {
        bigint id PK
        uuid user_id FK
        uuid session_id FK
        varchar query_role "搜索的职位关键词"
        varchar query_location
        jsonb raw_results "搜索API原始结果"
        varchar user_decision "approve | reject | modified"
        varchar saved_file_path
        timestamptz created_at
    }

    resume_snapshots {
        bigint id PK
        uuid user_id FK
        text resume_text "简历原文"
        text ai_feedback "AI评审意见"
        integer version
        timestamptz created_at
    }
```

---

## 三、每张表的设计理由

### 1. `users` — 用户表
**为什么需要**：当前项目用 `session_id`（前端随机生成）做隔离，没有真正的用户概念。加了用户表后：
- 同一用户可以有多个会话（切换话题不丢历史）
- 可对接 Supabase Auth / OAuth，实现真正的登录
- `last_active_at` 用于活跃用户统计

### 2. `sessions` — 会话表
**为什么需要**：把会话从 Checkpoint 的 `thread_id` 提升为一等公民：
- `title` 由 LLM 自动摘要（类似 ChatGPT 的侧边栏标题）
- `session_type` 标记会话主题，方便用户在前端查看"我的面试练习 / 我的求职记录"
- 与 LangGraph 的 `thread_id` 通过 `thread_id` 字段关联，**不破坏现有 Checkpointer 机制**

### 3. `messages` — 消息表（独立于 Checkpoint）
**为什么需要**：Checkpoint 里的 messages 是序列化的 blob，无法直接 SQL 查询。独立存储后：
- 可以做全文搜索（"用户之前问过什么关于 RAG 的问题？"）
- 可以统计 token 用量（通过 `metadata` 字段）
- 可以做消息级别的审计和合规检查
- **写入时机**：在每轮 `app.invoke()` 结束后，异步写入，不阻塞主流程

### 4. `user_profiles` — 用户画像表
**为什么需要**：当前画像存在 State 里，跟着 Checkpoint 走。问题是：
- 用户开新会话，画像从零开始
- 无法跨会话积累和查询

独立建表后，[update_profile](file:///d:/%E6%A1%8C%E9%9D%A2/Agent%E9%A1%B9%E7%9B%AE/Hello-Agents/Project/GenAI%20Career%20Assistant/nodes/profile.py#47-79) 节点在更新 State 的同时，也 upsert 到这张表。新会话启动时从表中加载已知画像注入 System Prompt。

### 5. `interview_records` — 面试记录表
**为什么需要**：面试练习是项目的核心功能，结构化记录后可以做：
- 薄弱点分析："你在系统设计类问题上平均得分 65，建议加强"
- 进步追踪：可视化展示用户历次面试评分趋势
- `qa_pairs` 用 JSONB 存原始问答，灵活且可查询

### 6. `job_search_logs` — 求职搜索日志表
**为什么需要**：记录每次搜索行为和用户的 HITL 审核决策：
- 可分析用户偏好趋势（"最近3次都在搜北京的NLP岗位"）
- 保存 `user_decision` 审核结果，作为 HITL 设计的数据证据

### 7. `resume_snapshots` — 简历快照表
**为什么需要**：简历迭代是高价值数据：
- 版本化存储：用户每次上传/修改简历都留存
- 可做前后对比："第3版 vs 第1版改了什么"

---

## 四、与 LangGraph Checkpointer 的共存策略

> [!IMPORTANT]
> **不替换 Checkpointer，而是与之并行。**

```
┌─────────────────────────────────────────────────┐
│                用户请求                          │
│                  │                              │
│          ┌───────▼────────┐                     │
│          │  LangGraph 图   │                    │
│          │  (Checkpointer) │──── 自动管理 ──── PostgresSaver 表   │
│          └───────┬────────┘    (checkpoint*)    │
│                  │                              │
│         图执行完毕，END 节点后                    │
│                  │                              │
│          ┌───────▼────────┐                     │
│          │  after_invoke   │                    │
│          │  回调/中间件     │──── 异步写入 ──── 自建业务表         │
│          └────────────────┘    (messages,       │
│                                interview_records│
│                                job_search_logs) │
└─────────────────────────────────────────────────┘
```

**原则**：
- Checkpointer 仍然是图执行和状态恢复的唯一来源（保证 LangGraph 语义不被破坏）
- 业务表是**异步旁路写入**，职责是提供查询和分析能力
- 即使业务表写入失败，也不影响核心对话流程

---

## 五、关键索引设计

```sql
-- 高频查询：用户的所有会话，按最近更新排序
CREATE INDEX idx_sessions_user_active ON sessions(user_id, updated_at DESC);

-- 消息全文搜索（PostgreSQL 内置 tsvector）
CREATE INDEX idx_messages_content_fts ON messages USING gin(to_tsvector('english', content));

-- 面试记录按用户+时间查询
CREATE INDEX idx_interview_user_time ON interview_records(user_id, created_at DESC);

-- 按 thread_id 快速关联 LangGraph checkpoint
CREATE UNIQUE INDEX idx_sessions_thread ON sessions(thread_id);
```

---

## 六、面试话术总结

> "我会在保留 LangGraph Checkpointer 的前提下，**旁路构建 7 张业务表**：`users`、`sessions`、`messages`、`user_profiles`、`interview_records`、`job_search_logs`、`resume_snapshots`。Checkpointer 负责图执行的状态恢复，业务表负责结构化查询和数据分析。写入策略是异步旁路，不阻塞主流程。这样既不破坏 LangGraph 的执行语义，又能支撑用户画像跨会话积累、面试薄弱点分析、求职行为追踪等业务需求。"
