"""
fc_router.py — 微调模型推理封装

将微调后的小模型封装为 Function Call Router，可替代原项目中的 LLM 分类路由层。

用法:
  from inference.fc_router import FCRouter

  router = FCRouter("models/fc_router_3b")
  result = router.route("帮我搜一下上海的 AI 岗位")
  # result = {"name": "search_jobs", "arguments": {"role": "AI", "location": "上海"}}
"""

import json
import re
from pathlib import Path
from typing import Optional


class FCRouter:
    """Function Call 路由器 — 用微调小模型替代 Prompt Engineering 路由。"""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Args:
            model_path: 合并后的模型路径 (HuggingFace 格式)
            device: "auto" | "cuda" | "cpu"
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # 加载工具 schema (用于 system prompt)
        schema_path = Path(__file__).parent.parent / "data" / "tools_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            self.tools_schema = json.load(f)
        self.tools_str = json.dumps(self.tools_schema, ensure_ascii=False, indent=2)

        self.system_prompt = (
            "You are a GenAI Career Assistant. You have access to the following tools:\n\n"
            f"<tools>\n{self.tools_str}\n</tools>\n\n"
            "For each user request, decide whether to call a tool or respond directly.\n"
            "If calling a tool, output ONLY: <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n"
            "If the user's request is outside your capabilities, politely decline.\n"
            "If the user's request is missing required parameters, ask a follow-up question."
        )

    def route(self, user_input: str, chat_history: list = None) -> dict:
        """对用户输入做路由决策。

        Args:
            user_input: 用户消息
            chat_history: 历史消息列表 [{"role": "user/assistant", "content": "..."}]

        Returns:
            dict: {
                "type": "tool_call" | "text",
                "name": "工具名" (type=tool_call 时),
                "arguments": {...} (type=tool_call 时),
                "content": "文本回复" (type=text 时),
                "raw_output": "模型原始输出",
            }
        """
        import torch

        messages = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)

        # 解析输出
        parsed = self._parse_tool_call(raw_output)
        if parsed:
            return {
                "type": "tool_call",
                "name": parsed["name"],
                "arguments": parsed["arguments"],
                "raw_output": raw_output,
            }
        else:
            return {
                "type": "text",
                "content": raw_output,
                "raw_output": raw_output,
            }

    def route_to_node(self, user_input: str, chat_history: list = None) -> tuple[str, dict]:
        """路由到原项目的节点名称（兼容 LangGraph 工作流）。

        映射关系:
          search_jobs → job_search
          generate_resume → handle_resume_making
          get_interview_questions → interview_topics_questions
          start_mock_interview → mock_interview
          generate_tutorial → tutorial_agent
          ask_ai_question → ask_query_bot
          (无工具调用) → out_of_scope

        Returns:
            (node_name, route_result) — 同时返回详细路由结果，避免重复推理
        """
        TOOL_TO_NODE = {
            "search_jobs": "job_search",
            "generate_resume": "handle_resume_making",
            "get_interview_questions": "interview_topics_questions",
            "start_mock_interview": "mock_interview",
            "generate_tutorial": "tutorial_agent",
            "ask_ai_question": "ask_query_bot",
        }

        result = self.route(user_input, chat_history)
        if result["type"] == "tool_call":
            return TOOL_TO_NODE.get(result["name"], "out_of_scope"), result
        return "out_of_scope", result

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[dict]:
        """从模型输出中解析 tool_call JSON（支持嵌套参数）。"""
        # 优先匹配 <tool_call> 标签
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 兜底: 括号计数法提取完整 JSON
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[i:j+1])
                            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
        return None


# ── 演示 ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time as _time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    args = parser.parse_args()

    router = FCRouter(args.model)

    test_queries = [
        "帮我搜一下上海有什么 AI 相关的工作",
        "给我出 5 道 RAG 的面试题",
        "今天天气怎么样？",
        "写一篇关于 LangChain 的教程",
        "帮我做一份 AI 工程师的简历",
    ]

    for q in test_queries:
        start = _time.perf_counter()
        node, result = router.route_to_node(q)  # 单次推理同时拿到节点名和详情
        elapsed_ms = (_time.perf_counter() - start) * 1000

        print(f"\n{'─'*50}")
        print(f"输入: {q}")
        print(f"类型: {result['type']}")
        if result["type"] == "tool_call":
            print(f"工具: {result['name']}")
            print(f"参数: {result['arguments']}")
        else:
            print(f"回复: {result['content'][:100]}...")
        print(f"节点: {node}")
        print(f"延迟: {elapsed_ms:.1f} ms")
