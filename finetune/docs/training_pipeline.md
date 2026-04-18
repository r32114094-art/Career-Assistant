# Function Call 微调训练流水线

> **项目**: GenAI Career Assistant — FC Router Fine-Tuning  
> **目标**: 用微调 Qwen2.5-3B 替代 Prompt-based 路由层，实现低延迟 (~200ms) 高准确率工具调用  
> **训练策略**: SFT → DPO → GRPO 三阶段训练链路

---

## 一、全局架构

```
Phase 1: 数据工程 (本地)
  tools_schema.json (6个工具定义)
    → 01_generate_sft_data.py (Teacher 生成 ~800 条)
      → 01b_split_dataset.py (train/val/test 拆分)

Phase 2: SFT (AutoDL)
  train.jsonl → 02_train_sft.py (Qwen2.5-3B + QLoRA) → SFT Checkpoint

Phase 3: DPO (AutoDL)
  SFT Checkpoint → 03_generate_dpo_data.py (SFT 错误 + Teacher 合成)
    → 04_train_dpo.py (偏好对对齐) → DPO Checkpoint

Phase 4: GRPO (AutoDL)
  DPO Checkpoint → 05_train_grpo.py (奖励函数驱动优化) → GRPO Checkpoint

Phase 5: 部署 & 评估
  GRPO Checkpoint → 06_merge_and_export.py (LoRA 合并导出)
    → fc_router.py (推理封装) → evaluate.py (6 指标 + 消融实验)
```

---

## 二、目录结构与文件职责

```
finetune/
├── data/                              # ⬇️ 数据层
│   ├── tools_schema.json              # 📌 6 个工具的 JSON Schema 定义（全局基石）
│   ├── sft/                           # SFT 数据目录
│   │   ├── sft_simple_call.jsonl      # (生成) 各类别原始数据
│   │   ├── train.jsonl                # (拆分) 训练集 80%
│   │   ├── val.jsonl                  # (拆分) 验证集 10%
│   │   └── test.jsonl                 # (拆分) 测试集 10%
│   ├── dpo/
│   │   └── preferences.jsonl          # (生成) chosen/rejected 偏好对
│   └── grpo/
│       └── prompts.jsonl              # (复用) val.jsonl 的子集作为 GRPO prompt
│
├── configs/                           # ⬇️ 配置层
│   ├── sft_config.yaml                # SFT 超参: LoRA rank=64, lr=2e-4, epoch=3
│   ├── dpo_config.yaml                # DPO 超参: beta=0.1, lr=5e-5, epoch=2
│   └── grpo_config.yaml               # GRPO 超参: group=8, kl=0.04, lr=1e-5
│
├── scripts/                           # ⬇️ 执行层（按编号顺序运行）
│   ├── 01_generate_sft_data.py        # 步骤1: Teacher 生成 SFT 数据
│   ├── 01b_split_dataset.py           # 步骤2: 合并 & 拆分 train/val/test
│   ├── 02_train_sft.py                # 步骤3: SFT 训练 (Unsloth)
│   ├── 03_generate_dpo_data.py        # 步骤4: 生成 DPO 偏好对
│   ├── 04_train_dpo.py                # 步骤5: DPO 训练
│   ├── 05_train_grpo.py               # 步骤6: GRPO 训练
│   └── 06_merge_and_export.py         # 步骤7: LoRA 合并导出
│
├── eval/                              # ⬇️ 评估层
│   ├── reward_function.py             # 🔑 多维度奖励函数（GRPO 训练 + 全阶段评估共用）
│   ├── evaluate.py                    # 统一评估入口（本地模型 / API 都支持）
│   └── results/                       # 消融实验结果 (E1~E6)
│
├── inference/                         # ⬇️ 集成层
│   └── fc_router.py                   # 微调模型推理封装 → 可直接替换原项目路由
│
├── docs/                              # ⬇️ 文档
│   └── training_pipeline.md           # 本文件
│
└── requirements.txt                   # 依赖说明
```

---

## 三、执行顺序

### 🏠 阶段一：本地执行（Windows 机器）

| 序号 | 命令 | 做什么 | 产出 |
|-----|------|--------|------|
| **①** | `python scripts/01_generate_sft_data.py --all` | 用 DeepSeek-V3 生成 8 类训练数据 | `data/sft/sft_*.jsonl` (~800 条) |
| **②** | `python scripts/01b_split_dataset.py` | 分层抽样拆分 train/val/test | `train.jsonl` / `val.jsonl` / `test.jsonl` |
| **③** | `copy data\sft\val.jsonl data\grpo\prompts.jsonl` | 准备 GRPO prompt 数据 | `data/grpo/prompts.jsonl` |

> **注意**: 步骤①②③在本地完成，它们只调用 DeepSeek API 生成数据，不需要 GPU。
> 完成后把整个 `finetune/` 目录上传到 AutoDL。

**前置条件**:
```bash
# 1. 安装依赖
pip install openai python-dotenv pyyaml

# 2. 创建 .env 文件
# DEEPSEEK_API_KEY=你的密钥
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
# DEEPSEEK_MODEL=deepseek-chat
```

---

### 🖥️ 阶段二：AutoDL 环境准备

```bash
# SSH 到 AutoDL 服务器后
pip install unsloth
pip install trl>=0.14.0 transformers>=4.46.0 datasets>=3.0.0
pip install peft>=0.13.0 bitsandbytes>=0.44.0 accelerate>=1.0.0
pip install openai python-dotenv pyyaml numpy
```

---

### 🖥️ 阶段三：GPU 训练（AutoDL）

| 序号 | 命令 | 做什么 | 耗时估算 | 产出 |
|-----|------|--------|---------|------|
| **④** | `python scripts/02_train_sft.py` | SFT 训练 | ~30-60 min | `checkpoints/sft_qwen25_3b/` |
| **⑤** | `python eval/evaluate.py --model checkpoints/sft_qwen25_3b --test-data data/sft/test.jsonl --output eval/results/e1_sft_only.json --latency` | 评估 SFT → E1 | ~5 min | `results/e1_sft_only.json` |
| **⑥** | `python scripts/03_generate_dpo_data.py --mode synthetic --count 400` | 生成 DPO 偏好对 | ~10 min | `data/dpo/preferences.jsonl` |
| **⑦** | `python scripts/04_train_dpo.py` | DPO 训练 | ~20-40 min | `checkpoints/dpo_qwen25_3b/` |
| **⑧** | `python eval/evaluate.py --model checkpoints/dpo_qwen25_3b --test-data data/sft/test.jsonl --output eval/results/e2_sft_dpo.json --latency` | 评估 DPO → E2 | ~5 min | `results/e2_sft_dpo.json` |
| **⑨** | `python scripts/05_train_grpo.py` | GRPO 训练 | ~40-80 min | `checkpoints/grpo_qwen25_3b/` |
| **⑩** | `python eval/evaluate.py --model checkpoints/grpo_qwen25_3b --test-data data/sft/test.jsonl --output eval/results/e3_full_pipeline.json --latency` | 评估 GRPO → E3 | ~5 min | `results/e3_full_pipeline.json` |

---

### 🖥️ 阶段四：导出与验收

| 序号 | 命令 | 做什么 | 产出 |
|-----|------|--------|------|
| **⑪** | `python scripts/06_merge_and_export.py --checkpoint checkpoints/grpo_qwen25_3b --output models/fc_router_3b` | LoRA 合并导出 | `models/fc_router_3b/` |
| **⑫** | `python inference/fc_router.py --model models/fc_router_3b` | 端到端验证 | 控制台输出路由结果 + 延迟 |
| **⑬** | `python eval/evaluate.py --api deepseek --test-data data/sft/test.jsonl --output eval/results/e5_deepseek_zero_shot.json` | (可选) DeepSeek 基线对比 | `results/e5.json` |

---

## 四、数据格式规范

### SFT 数据格式

每条 JSONL 行的结构:
```json
{
  "messages": [
    {"role": "system", "content": "You are a GenAI Career Assistant. Tools: ..."},
    {"role": "user", "content": "帮我搜上海的 AI 岗位"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"search_jobs\",\"arguments\":{\"role\":\"AI\",\"location\":\"上海\"}}</tool_call>"}
  ],
  "metadata": {
    "category": "simple_call",
    "raw": {"user": "...", "tool_call": {...}}
  }
}
```

### DPO 数据格式

```json
{
  "prompt": "帮我做一份简历",
  "chosen": "<tool_call>{\"name\":\"generate_resume\",\"arguments\":{\"target_role\":\"AI工程师\"}}</tool_call>",
  "rejected": "<tool_call>{\"name\":\"search_jobs\",\"arguments\":{\"role\":\"AI工程师\",\"location\":\"全国\"}}</tool_call>"
}
```

### GRPO 数据格式

GRPO 不需要标签，只需 prompt + 奖励函数:
```
prompt → 模型生成 8 个候选 → 奖励函数打分 → 组内排序 → 更新策略
```

---

## 五、奖励函数核心机制

`eval/reward_function.py` 同时服务于 **GRPO 训练**和**全阶段评估**。

### 五维度打分 (总分 0~1)

| 维度 | 权重 | 检查内容 |
|------|------|---------|
| 格式合规 | 0.2 | 能否解析出 `<tool_call>` JSON |
| 工具选择 | 0.3 | 工具名是否正确 |
| 必填参数 | 0.3 | required 参数值是否匹配 |
| 可选参数 | 0.1 | optional 参数的准确性 |
| 无幻觉 | 0.1 | 是否存在 schema 外的假参数 |

### 自测结果

| Case | 场景 | 得分 | 解释 |
|------|------|------|------|
| 1 | 完美调用 | **1.00** | 所有维度满分 |
| 2 | 工具选错 | **0.20** | 只有格式分，工具错后直接截断 |
| 3 | 正确拒绝 | **1.00** | 不该调用且没调用 = 满分 |
| 4 | 应拒绝但调用了 | **0.00** | 最严重的错误类型 |
| 5 | 参数幻觉 (salary) | **0.90** | 扣了 0.1 的无幻觉分 |

---

## 六、与原项目的集成

`inference/fc_router.py` 提供了与原项目 LangGraph 工作流的桥接:

```python
# 原项目的路由方式 (router.py + categorize 节点):
#   用户输入 → DeepSeek-V3 分类 → route_query() → 条件边
#
# 微调后的路由方式:
#   用户输入 → FCRouter.route_to_node() → 直接返回节点名

TOOL_TO_NODE = {
    "search_jobs":              "job_search",
    "generate_resume":          "handle_resume_making",
    "get_interview_questions":  "interview_topics_questions",
    "start_mock_interview":     "mock_interview",
    "generate_tutorial":        "tutorial_agent",
    "ask_ai_question":          "ask_query_bot",
}
```

> **架构关键**: 微调模型**只替代分类/路由层**，不替代每个 Agent 节点内部的业务 LLM。
> 节点内部（教程生成、简历制作等）仍由 DeepSeek-V3 完成——这是 **"小模型做路由，大模型做生成"** 的经典架构。

---

## 七、消融实验设计

| 实验 | 模型 | 训练 | 对比目的 |
|-----|------|------|---------| 
| E1 | Qwen2.5-3B | SFT only | 基线：纯 SFT 能到什么水平 |
| E2 | Qwen2.5-3B | SFT + DPO | DPO 的增量价值 |
| E3 | Qwen2.5-3B | SFT + DPO + GRPO | 完整三阶段 |
| E4 | Qwen2.5-7B | SFT + DPO + GRPO | 模型规模的影响 |
| E5 | DeepSeek-V3 | zero-shot FC | 对照组：大模型无微调 |
| E6 | DeepSeek-V3 | few-shot FC | 对照组：大模型 few-shot |

> **核心卖点**: E3 vs E5 = 3B 微调 vs 671B 大模型的性价比论证。

---

## 八、关键超参汇总

### SFT 阶段

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-3B-Instruct | 3B 参数量 |
| LoRA Rank | 64 | 中等复杂度 |
| LoRA Alpha | 128 | alpha/rank = 2 |
| 量化 | 4-bit (QLoRA) | 适配单卡显存 |
| 学习率 | 2e-4 | SFT 标准 |
| Epoch | 3 | 数据量 ~640 条 |
| 有效 Batch | 16 | 4 × 4 (grad_accum) |

### DPO 阶段

| 参数 | 值 | 说明 |
|------|-----|------|
| Beta | 0.1 | DPO KL 系数 |
| 学习率 | 5e-5 | 比 SFT 小一个量级 |
| Epoch | 2 | 偏好对 ~400 条 |

### GRPO 阶段

| 参数 | 值 | 说明 |
|------|-----|------|
| Group Size | 8 | 每个 prompt 生成 8 个候选 |
| Temperature | 0.7 | 保证多样性 |
| KL Penalty | 0.04 | 防止偏离太远 |
| 学习率 | 1e-5 | 最保守 |

---

## 九、代码审查修复记录

五轮审查共发现并修复 **20 个问题**:

| # | 文件 | 严重度 | 问题 | 修复方案 |
|---|------|-------|------|---------|
| 1 | `01_generate_sft_data.py` | 🔴 BUG | API 连续失败死循环 | `MAX_RETRIES_PER_BATCH=5` |
| 2 | `01_generate_sft_data.py` | 🟡 | Teacher 输出含脏数据 | 逐条结构校验 |
| 3 | `01_generate_sft_data.py` | 🟡 | 未校验 API Key | 启动前置校验 |
| 4 | `01b_split_dataset.py` | 🔴 BUG | 随机拆分丢失小类别 | 分层抽样 |
| 5 | `01b_split_dataset.py` | 🟡 | 路径相对 CWD 解析 | 统一 `finetune_dir` 基准 |
| 6 | `evaluate.py` | 🔴 BUG | `reward_function` 导入失败 | `sys.path.insert` |
| 7 | `evaluate.py` | 🟡 | `measure_latency` 重复加载模型 | 复用已加载对象 |
| 8 | `reward_function.py` | 🔴 BUG | 正则无法解析嵌套 JSON | 括号计数法 |
| 9 | `fc_router.py` | 🟡 | 演示中重复推理 | `route_to_node` 返回元组 |
| 10 | `fc_router.py` | 🔴 BUG | `_parse_tool_call` 同 #8 | 括号计数法 |
| 11 | `03_generate_dpo_data.py` | 🔴 BUG | 缺 API Key 校验 + 死循环 | 前置校验 + 重试上限 |
| 12 | `05_train_grpo.py` | 🔴 BUG | multi_turn 类 ground truth 误判 | 增加 `tool_call_or_response` 分支 |
| 13 | `evaluate.py` | 🔴 BUG | multi_turn 类评估 ground truth 丢失 | 同步增加 `tool_call_or_response` 分支 |
| 14 | `06_merge_and_export.py` | 🔴 BUG | `merge_and_unload()` 非 Unsloth API | 改用 `save_pretrained_merged()` |
| 15 | `02_train_sft.py` | 🔴 **CRASH** | TRL≥1.0 中 SFTTrainer 构造参数已迁移 | 迁移至 `SFTConfig` |
| 16 | `05_train_grpo.py` | 🔴 **CRASH** | `completions` 格式 `list[list[dict]]` 非 `list[str]` | 从 `completion[0]["content"]` 提取 |
| 17 | `05_train_grpo.py` | 🔴 **CRASH** | ground truth 闭包索引在多 batch 下偏移 | 改用 dataset 列 + `**kwargs` 注入 |
| 18 | `04_train_dpo.py` | 🔴 BUG | DPO prompt 缺少 system prompt | 转为 conversational 格式 |
| 19 | `05_train_grpo.py` | 🔴 **CRASH** | prompt 列存预格式化文本致双重 chat_template | 改存 raw message list |
| 20 | `05_train_grpo.py` | 🟡 | `tokenizer` 参数名已更新为 `processing_class` | 改用 `processing_class=tokenizer` |

---

## 十、预计资源消耗

| 资源 | 估算 |
|------|------|
| DeepSeek API | ~800 条 SFT + ~400 条 DPO ≈ ¥5-15 |
| AutoDL GPU 时长 | ~2-4 小时 (A100/A800) |
| 总耗时 | 本地 30 min + GPU 2-4 h |
