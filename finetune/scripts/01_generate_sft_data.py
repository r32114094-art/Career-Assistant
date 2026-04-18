"""
01_generate_sft_data.py — 用 Teacher 模型批量生成 SFT 训练数据

使用 DeepSeek-V3 作为 Teacher，生成 8 类 Function Call 训练样本:
  1. 单工具简单调用 (simple_call)
  2. 含可选参数调用 (optional_params)
  3. 参数不完整追问 (ask_followup)
  4. 拒绝调用 (reject)
  5. 上下文依赖多轮 (multi_turn)
  6. 多意图拆解 (multi_intent)
  7. 边界消歧 (disambiguation)
  8. 中英文混合 (bilingual)

用法:
  python scripts/01_generate_sft_data.py --category simple_call --count 30 --output data/sft/raw/
  python scripts/01_generate_sft_data.py --all --output data/sft/raw/
"""

import os
import json
import argparse
import time
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Windows 终端 GBK 编码兼容：强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MAX_RETRIES_PER_BATCH = 5  # 单批次最大重试次数，防止死循环

# ── 配置 ──────────────────────────────────────────────

TEACHER_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEACHER_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TEACHER_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# 加载工具定义
SCRIPT_DIR = Path(__file__).parent
TOOLS_SCHEMA_PATH = SCRIPT_DIR.parent / "data" / "tools_schema.json"

with open(TOOLS_SCHEMA_PATH, "r", encoding="utf-8") as f:
    TOOLS_SCHEMA = json.load(f)

TOOLS_SCHEMA_STR = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)

# 系统提示词模板（训练时使用的真实系统提示）
SYSTEM_PROMPT = """You are a GenAI Career Assistant. You have access to the following tools:

<tools>
{tools}
</tools>

For each user request, decide whether to call a tool or respond directly.
If calling a tool, output ONLY: <tool_call>{{"name":"tool_name","arguments":{{...}}}}</tool_call>
If the user's request is outside your capabilities (not about AI careers, learning, interviews, jobs, or resumes), politely decline.
If the user's request is missing required parameters for a tool, ask a follow-up question to get the missing information."""

# ── 类别定义 ─────────────────────────────────────────

CATEGORIES = {
    "simple_call": {
        "count": 180,
        "description": "单工具简单调用：意图明确，参数齐全，模型应直接输出 tool_call",
        "instruction": """生成用户查询和对应的工具调用对。要求:
- 用户查询自然流畅，像真实用户的表述
- 每个查询明确对应一个工具，且提供了所有必填参数
- 覆盖全部 6 个工具，大致均匀分布
- 表述风格多样：口语化/书面化/简洁/详细
- 中文为主，适当混入英文技术术语

每条数据格式:
{{"user": "用户查询", "tool_call": {{"name": "工具名", "arguments": {{参数字典}}}}}}

注意: arguments 中只包含用户明确提供的参数，不要猜测用户未说的内容。""",
    },
    "optional_params": {
        "count": 120,
        "description": "含可选参数：用户提供了可选参数，模型需正确提取",
        "instruction": """生成包含可选参数的工具调用数据。要求:
- 用户查询中自然地包含可选参数信息（如难度、数量、类型等）
- 模型需正确提取必填和可选参数
- 覆盖各种参数类型：enum (difficulty, job_type, level), integer (count), boolean (include_code), array (skills, focus_areas)

每条数据格式:
{{"user": "用户查询", "tool_call": {{"name": "工具名", "arguments": {{包含可选参数的字典}}}}}}""",
    },
    "ask_followup": {
        "count": 100,
        "description": "参数不完整追问：缺少必填参数，模型应追问而非调用",
        "instruction": """生成缺少必填参数的用户查询，模型应追问而非调用工具。要求:
- 用户查询缺少至少一个 required 参数
- 例如 search_jobs 缺 location、generate_resume 缺 target_role
- 模型回复应是友好的追问，明确指出缺少什么信息
- 不要包含 <tool_call> 标签

每条数据格式:
{{"user": "用户查询", "assistant": "追问回复（不含 tool_call）"}}""",
    },
    "reject": {
        "count": 100,
        "description": "拒绝调用：完全超范围请求，模型应友好拒绝",
        "instruction": """生成完全超出 GenAI 职业助手范围的用户查询，模型应友好拒绝。要求:
- 查询主题包括：天气、美食、旅游、数学计算、翻译、游戏、八卦、健康等
- 模型回复应友好但明确地说明自己的能力范围
- 不包含任何 <tool_call> 标签
- 回复中简要提及支持的功能方向

每条数据格式:
{{"user": "用户查询", "assistant": "友好拒绝回复"}}""",
    },
    "multi_turn": {
        "count": 80,
        "description": "多轮上下文：结合对话历史做出正确决策",
        "instruction": """生成多轮对话场景，模型需根据上下文做出正确工具调用。要求:
- 包含 1-3 轮历史对话
- 当前用户消息可能是简短的（如"继续"、"好的，那搜一下"、"换个方向"）
- 模型必须综合历史上下文理解真实意图
- 场景包括：话题延续、意图切换、补充参数

每条数据格式:
{{"history": [
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
], "user": "当前用户消息", "tool_call_or_response": "工具调用或文本回复"}}""",
    },
    "multi_intent": {
        "count": 60,
        "description": "多意图拆解：一句话包含多个意图，模型应处理主要意图",
        "instruction": """生成包含多个意图的用户查询。要求:
- 用户一句话中包含 2 个或以上意图
- 模型应优先处理第一个/最主要的意图
- 例如："帮我搜岗位然后做简历" → 先 search_jobs

每条数据格式:
{{"user": "用户查询", "tool_call": {{"name": "主意图工具", "arguments": {{...}}}}, "note": "为什么选择这个工具"}}""",
    },
    "disambiguation": {
        "count": 80,
        "description": "边界消歧：模糊表述需正确归类",
        "instruction": """生成容易混淆工具选择的模糊查询。要求:
- 查询在两个工具之间模棱两可
- 例如："AI 会取代程序员吗" → ask_ai_question（不是面试题）
- "RAG 有什么考点" → get_interview_questions（不是技术问答）
- 每条标注正确的工具和选择理由

每条数据格式:
{{"user": "用户查询", "tool_call": {{"name": "正确工具", "arguments": {{...}}}}, "confusable_with": "容易混淆的工具", "reason": "选择理由"}}""",
    },
    "bilingual": {
        "count": 80,
        "description": "中英文混合：系统需正确处理双语输入",
        "instruction": """生成中英文混合的用户查询。要求:
- 包含纯英文、纯中文、中英混合三种形式
- 技术术语可以用英文，日常表达用中文
- 模型照常做工具调用，参数值保持用户的原始语言

每条数据格式:
{{"user": "用户查询", "tool_call": {{"name": "工具名", "arguments": {{...}}}}}}""",
    },
}


# ── 数据生成 ─────────────────────────────────────────

def generate_batch(
    client: OpenAI,
    category: str,
    batch_size: int = 20,
) -> list[dict]:
    """调用 Teacher 模型生成一批训练数据。"""
    cat_info = CATEGORIES[category]

    prompt = f"""你是一名高质量训练数据生成专家。请为 GenAI Career Assistant 的 Function Call 微调生成训练数据。

## 工具定义
{TOOLS_SCHEMA_STR}

## 数据类别: {cat_info['description']}

## 生成要求
{cat_info['instruction']}

## 额外约束
- 生成 {batch_size} 条不重复的数据
- 输出为 JSON 数组，每个元素是一条数据
- 确保数据多样性和自然度
- 只输出 JSON 数组，不要有其他文本

请开始生成:"""

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的 AI 训练数据生成引擎。只输出合法的 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=8192,
        )
        content = response.choices[0].message.content.strip()

        # 清理 markdown 代码块标记
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]

        # 校验每条数据的基本结构
        validated = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # 必须有 user 字段
            if "user" not in item and "history" not in item:
                continue
            # 必须有输出字段
            if not any(k in item for k in ("tool_call", "assistant", "tool_call_or_response")):
                continue
            # tool_call 校验
            if "tool_call" in item:
                tc = item["tool_call"]
                if not isinstance(tc, dict) or "name" not in tc:
                    continue
                if "arguments" not in tc:
                    tc["arguments"] = {}
            validated.append(item)

        return validated

    except Exception as e:
        print(f"  ⚠️ 生成失败: {e}")
        return []


def format_to_sft(raw_item: dict, category: str) -> dict:
    """将 Teacher 生成的原始数据转换为 SFT 训练格式 (messages 列表)。"""
    system_content = SYSTEM_PROMPT.format(tools=TOOLS_SCHEMA_STR)
    messages = [{"role": "system", "content": system_content}]

    # 多轮对话：添加历史
    if "history" in raw_item:
        for turn in raw_item["history"]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # 用户消息
    user_content = raw_item.get("user", "")
    if user_content:
        messages.append({"role": "user", "content": user_content})

    # 助手回复
    if "tool_call" in raw_item:
        tc = raw_item["tool_call"]
        assistant_content = f'<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>'
        messages.append({"role": "assistant", "content": assistant_content})
    elif "assistant" in raw_item:
        messages.append({"role": "assistant", "content": raw_item["assistant"]})
    elif "tool_call_or_response" in raw_item:
        messages.append({"role": "assistant", "content": raw_item["tool_call_or_response"]})

    # 构建完整训练样本（带元数据）
    return {
        "messages": messages,
        "metadata": {
            "category": category,
            "raw": raw_item,
        },
    }


def build_ground_truth(raw_item: dict) -> dict:
    """从原始数据提取评估用的 ground truth。"""
    if "tool_call" in raw_item:
        tc = raw_item["tool_call"]
        # 分离 required 和 optional（需要 schema 信息，这里简化处理）
        return {
            "action": "call",
            "name": tc["name"],
            "required_params": tc.get("arguments", {}),
            "optional_params": {},
        }
    elif "assistant" in raw_item:
        content = raw_item["assistant"]
        if "?" in content or "？" in content:
            return {"action": "ask_followup"}
        else:
            return {"action": "reject"}
    return {"action": "reject"}


# ── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 SFT 训练数据")
    parser.add_argument("--category", type=str, choices=list(CATEGORIES.keys()),
                        help="指定生成的数据类别")
    parser.add_argument("--all", action="store_true", help="生成所有类别")
    parser.add_argument("--count", type=int, default=None, help="覆盖默认样本数")
    parser.add_argument("--batch-size", type=int, default=20, help="每批生成数量")
    parser.add_argument("--output", type=str, default="data/sft/", help="输出目录")
    args = parser.parse_args()

    # API 密钥校验
    if not TEACHER_API_KEY:
        print("❌ 未设置 DEEPSEEK_API_KEY 环境变量。请在 .env 文件中配置。")
        return

    client = OpenAI(api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL)
    output_dir = SCRIPT_DIR.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    categories_to_generate = list(CATEGORIES.keys()) if args.all else [args.category]

    for category in categories_to_generate:
        if category is None:
            print("请指定 --category 或 --all")
            return

        target_count = args.count or CATEGORIES[category]["count"]
        print(f"\n{'='*60}")
        print(f"📝 生成类别: {category} (目标: {target_count} 条)")
        print(f"{'='*60}")

        all_raw = []
        all_sft = []

        consecutive_failures = 0
        while len(all_raw) < target_count:
            remaining = target_count - len(all_raw)
            batch_size = min(args.batch_size, remaining)

            print(f"  🔄 批次 {len(all_raw)//args.batch_size + 1}: 生成 {batch_size} 条...")
            batch = generate_batch(client, category, batch_size)

            if batch:
                all_raw.extend(batch)
                consecutive_failures = 0
                print(f"  ✅ 本批获得 {len(batch)} 条，累计 {len(all_raw)}/{target_count}")
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_RETRIES_PER_BATCH:
                    print(f"  ❌ 连续 {MAX_RETRIES_PER_BATCH} 次生成失败，跳过剩余部分")
                    break
                print(f"  ⚠️ 本批生成失败 ({consecutive_failures}/{MAX_RETRIES_PER_BATCH})，重试中...")

            time.sleep(1)  # 限速

        # 转换为 SFT 格式
        for item in all_raw:
            sft_item = format_to_sft(item, category)
            all_sft.append(sft_item)

        # 保存
        raw_path = output_dir / f"raw_{category}.jsonl"
        sft_path = output_dir / f"sft_{category}.jsonl"

        with open(raw_path, "w", encoding="utf-8") as f:
            for item in all_raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(sft_path, "w", encoding="utf-8") as f:
            for item in all_sft:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  💾 原始数据: {raw_path}")
        print(f"  💾 SFT 数据: {sft_path}")

    print(f"\n{'='*60}")
    print(f"✅ 全部生成完成")


if __name__ == "__main__":
    main()
