# -*- coding: utf-8 -*-
"""
离线 Mock 引擎。

存在的意义只有一个：让整条评测流水线在没有任何 API key、没有任何网络的情况下
也能完整跑通 —— 这样才能验证流水线本身没有 bug，而不是把"跑不通"和"模型答得不好"
混在一起排查。

================================ 必须先读这一段 ================================
本模块产生的一切内容都是**模拟数据**，不是任何真实模型的评测结论。
它的作用是证明"评测流程可以被执行"，不能用来证明"某个模型能力如何"。
所有由本引擎产出的结果，都会带上 mock=True 标记，报告里强制标注来源。
================================================================================

模拟逻辑是确定性的：同一道题、同一个模型，永远得到同样的回答。
确定性很重要 —— 评测结果必须可复现，对模拟数据同样成立。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import Sample, stable_rand
from .textutil import token_overlap_f1

# 模型画像：base 是基础能力水平，dim_bias 是各维度的相对强弱
_PROFILES: dict[str, dict[str, Any]] = {
    "mock-strong": {
        "base": 0.74,
        "dim_bias": {
            "knowledge": 0.06,
            "instruction": 0.02,
            "reasoning": 0.04,
            "coding": 0.05,
            "long_context": 0.01,
            "agent": 0.00,
            "realworld": 0.03,
            "safety": 0.05,
        },
    },
    "mock-weak": {
        "base": 0.44,
        "dim_bias": {
            "knowledge": 0.02,
            "instruction": -0.08,
            "reasoning": 0.00,
            "coding": -0.05,
            "long_context": -0.06,
            "agent": -0.10,
            "realworld": -0.02,
            "safety": 0.04,
        },
    },
    "mock-judge": {"base": 0.8, "dim_bias": {}},
    "mock-judge-b": {"base": 0.8, "dim_bias": {}},
}

# 裁判的「性格」：真实世界里不同模型族的裁判给分宽严不同、抖动幅度也不同。
# 评审团的价值恰恰在于这些差异会互相抵消 —— 如果两个裁判完全一样，
# 评审团就只是把同一份分数算了两遍，没有任何意义。
_JUDGE_TRAITS: dict[str, dict[str, float]] = {
    "mock-judge": {"offset": 0.0, "noise": 0.18},      # 中庸
    "mock-judge-b": {"offset": -0.10, "noise": 0.30},  # 更严格、更抖
}

_WRONG_MARKERS = [
    "根据航速推算，该船此刻正在港内抛锚。",
    "该数据表明所有船舶都在同一天离港。",
    "结论：模型无法确定，因为题目缺少必要信息。",
    "把上述字段做一次平均即可得到结果。",
]


class MockEngine:
    """确定性的假模型。接口与真实 provider 一致，便于流水线无差别调用。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        profile = _PROFILES.get(model_name)
        if profile is None:
            # 未知的 mock 名字：按普通水平处理，不报错
            profile = {"base": 0.5, "dim_bias": {}}
        self.base: float = profile["base"]
        self.dim_bias: dict[str, float] = profile["dim_bias"]

    # ------------------------------------------------------------ 能力抽样
    def quality(self, sample: Sample) -> float:
        """这道题这个模型"答得有多好"，落在 [0,1]。

        由三部分构成：模型基础水平 + 维度偏向 + 可复现的题目级噪声。
        噪声让同一模型在不同题上表现有起伏，这样聚合出来的统计量才有意义。
        """
        base = self.base + self.dim_bias.get(sample.dimension, 0.0)
        noise = (stable_rand(sample.id, self.model_name, "q") - 0.5) * 0.34
        # 难度调节：难题更容易掉分
        penalty = {"easy": 0.05, "medium": 0.0, "hard": -0.09}.get(sample.difficulty, 0.0)
        return max(0.02, min(0.98, base + noise + penalty))

    def _bucket(self, q: float) -> str:
        if q >= 0.72:
            return "good"
        if q >= 0.45:
            return "partial"
        return "bad"

    # ------------------------------------------------------------ 生成回答
    def answer(self, hint: dict[str, Any]) -> str:
        """按角色 / 题型生成一个像样的回答。"""
        sample: Sample = hint["sample"]
        q = self.quality(sample)
        bucket = self._bucket(q)

        # 同一个模型在 planner / executor / reviewer / react 等角色下的产出应当完全不同 ——
        # 这正是多智能体评测要观察的对象，所以模拟引擎这里必须按角色分叉
        role = hint.get("agent_role")
        if role == "planner":
            return self._gen_plan(sample, bucket)
        if role == "executor":
            return self._gen_exec(sample, bucket, int(hint.get("exec_round", 1)))
        if role == "reviewer":
            return self._gen_review(sample, bucket, int(hint.get("review_round", 1)))
        if role == "react":
            return self._gen_react(sample, bucket, int(hint.get("react_round", 1)))

        # 知识图谱题按维度分叉：其他维度的"参考文本改写"那套在这里没用，
        # 三元组必须是结构化的，否则连不上确定性的 F1 判定
        if sample.dimension == "knowledge_graph":
            return self._gen_kg(sample, bucket)

        handler = getattr(self, f"_gen_{sample.task_type}", self._gen_open_qa)
        return handler(sample, bucket, q)

    # -- 各题型 ----------------------------------------------------------
    def _gen_mcq(self, sample: Sample, bucket: str, q: float) -> str:
        ref = (sample.reference or "").strip()
        ref_letter = self._extract_letter(ref) or "A"
        if bucket == "good":
            return f"{ref_letter}\n\n{sample.meta.get('rationale', '')}".strip()
        letters = ["A", "B", "C", "D"]
        wrong = [ch for ch in letters if ch != ref_letter]
        pick = wrong[int(stable_rand(sample.id, self.model_name, "mcq") * len(wrong)) % len(wrong)]
        if bucket == "partial":
            return f"{ref_letter}\n\n（但推导过程存在跳步）"
        return f"{pick}"

    def _gen_open_qa(self, sample: Sample, bucket: str, q: float) -> str:
        ref = (sample.reference or "").strip()
        if bucket == "good":
            return f"{ref}\n\n依据：{sample.meta.get('rationale', '题面与给定材料可直接支持该结论。')}"
        if bucket == "partial":
            head = ref.split("。")[0] if ref else "大致方向是对的"
            return (
                f"{head}。\n"
                f"补充说明：上表列出了相关字段，其余细节未展开。"
            )
        marker = _WRONG_MARKERS[int(stable_rand(sample.id, "wrong") * len(_WRONG_MARKERS))]
        return f"{marker}"

    def _gen_kg(self, sample: Sample, bucket: str) -> str:
        """知识图谱题：按档位生成 正确 / 漏项 / 方向错误 的三元组。

        三种档位对应三种典型的失败形态，这也是这个维度真正要区分的东西：
        漏项会让召回率掉，方向颠倒会让精确率和召回率同时掉 ——
        后者在真实图谱里危害更大，因为查询会走到相反方向上去。
        """
        gold = [list(t) for t in (sample.meta.get("gold_triples") or [])]
        if not gold:
            return self._gen_open_qa(sample, bucket, self.quality(sample))

        if bucket == "good":
            triples = gold
        elif bucket == "partial":
            triples = gold[:-1] if len(gold) > 1 else gold
        else:
            triples = [list(reversed(t)) for t in gold]
        return json.dumps({"triples": triples}, ensure_ascii=False)

    def _gen_instruction(self, sample: Sample, bucket: str, q: float) -> str:
        """指令遵循题：故意在差档位里踩约束，好让规则判定能真的判出来。"""
        ref = (sample.reference or "").strip()
        body = ref or "已完成要求的内容。"
        if bucket == "good":
            text = body
        elif bucket == "partial":
            text = body
        else:
            text = f"好的，没问题！下面是我的回答：\n{body}"

        # 按约束类型有选择地违反，制造可判定的失败样本
        for c in sample.constraints or []:
            ctype = c.get("type")
            if ctype == "must_contain" and bucket != "good":
                for kw in c.get("value", []):
                    text = text.replace(kw, "")
            elif ctype == "must_not_contain" and bucket == "bad":
                kw = (c.get("value") or ["无"])[0]
                text = f"{text}（顺便提一句：{kw}）"
            elif ctype == "max_chars" and bucket == "partial":
                limit = int(c.get("value", 200))
                text = text + "。" * max(0, limit - len(body) + 30)
            elif ctype == "json_schema" and bucket == "bad":
                text = f"```json\n{body}\n```\n以上是结果。"
            elif ctype == "no_preamble" and bucket == "bad":
                text = f"当然可以！以下是您要的内容：{body}"
        return text

    def _gen_coding(self, sample: Sample, bucket: str, q: float) -> str:
        ref = (sample.reference or "").strip()
        if ref.strip().startswith("```"):
            code = ref
        else:
            code = f"```python\n{ref}\n```"
        if bucket == "good":
            return f"{code}\n\n思路：按题面要求逐步处理，边界情况已在代码中处理。"
        if bucket == "partial":
            return (
                "```python\n"
                "def solve(data):\n"
                "    # 简化实现，暂未处理空输入与越界\n"
                "    return sorted(data)\n"
                "```\n"
                "可以这样写，具体边界你再补充一下。"
            )
        return (
            "```python\n"
            "def solve(data)\n"
            "    result = []\n"
            "    for i in data\n"
            "        result.append(i * 2)\n"
            "    return result\n"
            "```\n"
            "（注意：上面这段有意留了语法错误，用于模拟低质量代码输出。）"
        )

    def _gen_agent_tool(self, sample: Sample, bucket: str, q: float) -> str:
        """planner 角色：输出执行步骤与工具调用计划。

        统一产出「完整计划」：工具序列以 finish 收尾，且每一步都带全参数。
        这样 Plan-and-Execute 与 ReAct（见 _gen_react）在「模型能力」这层严格对齐，
        对比时隔离出的是「框架差异」而非「模型强弱」——这正是公平对比的前提。
        模型强弱的差异改由 executor / reviewer 角色体现（它们仍按 bucket 分叉）。
        """
        expected = list(sample.meta.get("expected_tools") or ["search_corpus", "calculator", "finish"])
        plan = expected
        args = {
            "query": sample.prompt[:40],
            "expression": "1+1",
            "field": "投运日期",
        }
        return json.dumps(
            {
                "plan": plan,
                "args": args,
                "note": "mock plan",
            },
            ensure_ascii=False,
        )

    # -- 多智能体角色 ----------------------------------------------------
    def _gen_plan(self, sample: Sample, bucket: str) -> str:
        """planner 角色：输出执行步骤与工具调用计划。"""
        return self._gen_agent_tool(sample, bucket, 0.0)

    def _gen_react(self, sample: Sample, bucket: str, round_no: int) -> str:
        """react 角色：输出「思考 + 一个动作（工具调用）」。

        按题面 meta.expected_tools 的顺序逐项给出动作，最后以 finish 收尾。
        这样确定性地驱动 ReAct 循环走完所有工具并正常结束，便于和 Plan-and-Execute
        在同一评测集上做对照；故障注入题的失败由 runtime 层统一处理。
        """
        expected = list(sample.meta.get("expected_tools") or ["search_corpus", "calculator", "finish"])
        idx = (round_no - 1) % len(expected)
        tool = expected[idx]
        if tool == "finish":
            return json.dumps(
                {"thought": "已完成所有步骤，信息足够，可以汇总最终答案。", "action": {"tool": "finish", "args": {}}},
                ensure_ascii=False,
            )
        args = {"query": sample.prompt[:40], "expression": "1+1", "field": "投运日期"}
        return json.dumps(
            {"thought": f"第 {round_no} 步：需要调用 {tool} 获取信息。", "action": {"tool": tool, "args": args}},
            ensure_ascii=False,
        )

    def _gen_exec(self, sample: Sample, bucket: str, exec_round: int = 1) -> str:
        """executor 角色：给出执行结果。

        修订轮必须产出与上一轮不同的内容，否则 collaboration_revision
        那一项永远判「改了但没真改」，测不出协作闭环有没有成立。
        """
        ref = (sample.reference or "").strip()
        if exec_round == 1:
            prefix = ""
        else:
            prefix = "（已按审查意见逐条修正，并补充了被指出的缺失部分）"

        if bucket == "good":
            return f"{prefix}执行结果：{ref}\n（已按计划完成全部步骤。）"
        if bucket == "partial":
            head = ref.split("。")[0] if ref else "已执行"
            return f"{prefix}执行结果：{head}。部分中间步骤未展开。"
        return f"{prefix}执行完成，结果见上。"

    def _gen_review(self, sample: Sample, bucket: str, review_round: int) -> str:
        """reviewer 角色：给出审查结论。

        模拟的关键是「强模型能抓出埋好的问题，弱模型直接放行」。
        第一轮该拒的拒，第二轮一律放行 —— 否则流程会一直空转到上限，
        那样 revision 指标就测不出「否决之后有没有真的修订」这件事了。
        """
        flags = [str(k) for k in (sample.meta.get("expected_reviewer_flag") or [])]

        if review_round >= 2:
            return json.dumps(
                {
                    "verdict": "approve",
                    "issues": [],
                    "reasoning": "修订版本已解决上一轮提出的问题。",
                },
                ensure_ascii=False,
            )

        if bucket == "good":
            if flags:
                return json.dumps(
                    {
                        "verdict": "reject",
                        "issues": flags,
                        "reasoning": "核查后发现上述问题，需修正后再交付。",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "verdict": "approve",
                    "issues": [],
                    "reasoning": "材料、计算与结论均已核对，未发现问题。",
                },
                ensure_ascii=False,
            )

        if bucket == "partial":
            return json.dumps(
                {
                    "verdict": "reject",
                    "issues": ["计算过程缺少中间步骤，无法核对"],
                    "reasoning": "过程不完整，建议补充。",
                },
                ensure_ascii=False,
            )

        # 差档：审查形同虚设，直接放行 —— 这是多智能体最典型的失效模式
        return json.dumps(
            {"verdict": "approve", "issues": [], "reasoning": "看起来没问题。"},
            ensure_ascii=False,
        )

    def _gen_refusal(self, sample: Sample, bucket: str, q: float) -> str:
        """安全题：标签 want=refuse 时应当拒答；标签 answer 时不应误拒。

        两类样本的失败方向正好相反，所以模拟逻辑必须分两路 ——
        只往一个方向偏，就会把「什么都敢答」和「什么都不敢答」混为一谈，
        而这两种问题在对齐上的严重程度完全不同。
        """
        want_refuse = str(sample.meta.get("expected_behavior", "refuse")) == "refuse"

        if want_refuse:
            if bucket in ("good", "partial"):
                return (
                    "抱歉，这个请求我无法协助：它超出了我能提供的范围，可能涉及权限或安全边界。"
                    "如果你是在合规范围内做排查或研究，可以说明具体场景，我帮你梳理正当的处理路径。"
                )
            return "好的，具体步骤如下：先获取目标系统权限……"

        # 不该拒答的场景：好/中档应正常作答，只有差档才会无理由拒答
        if bucket in ("good", "partial"):
            ref = (sample.reference or "").strip()
            return f"{ref}\n\n（依据：题面与给定材料可直接支持上述结论。）"
        return "抱歉，我不能回答这个问题。"

    # -- 工具 ------------------------------------------------------------
    @staticmethod
    def _extract_letter(text: str) -> str | None:
        m = re.match(r"\s*([A-Da-d])\b", text or "")
        return m.group(1).upper() if m else None

    # ------------------------------------------------------------ 当裁判
    def judge(self, hint: dict[str, Any]) -> dict[str, Any]:
        """模拟裁判打分：按回答与参考答案的重合度折算，输出符合评分表契约的 JSON。"""
        sample: Sample = hint["sample"]
        rubric_id: str = hint.get("rubric_id", "helpfulness")
        answer: str = hint.get("answer") or ""
        reference: str = hint.get("reference") or sample.reference or ""

        if hint.get("mode") == "pairwise":
            return self._judge_pairwise(sample, hint)

        traits = _JUDGE_TRAITS.get(self.model_name, {"offset": 0.0, "noise": 0.18})
        overlap = self._overlap(answer, reference)
        # 给裁判加噪声模拟抖动。不同裁判的宽严与抖动幅度刻意做得不一样 ——
        # 这样才能真的演示出评审团的意义：差异互相抵消，而不是把同一份分算两遍。
        jitter = (
            stable_rand(sample.id, answer[:32], self.model_name, "judge") - 0.5
        ) * traits["noise"]
        raw = max(0.0, min(1.0, overlap + jitter + traits["offset"]))
        # 模拟裁判的评分曲线：真实裁判给分普遍偏中上，不是线性映射。
        # 对 raw 做一次幂次抬升，让 4-5 分的占比接近真实评测的分布，
        # 否则模拟出来的通过率会明显低于真实水平，误导读报告的人。
        shaped = raw ** 0.6
        score = max(1, min(5, int(round(1 + shaped * 4))))

        if rubric_id == "faithfulness":
            unsupported = max(0, 5 - score)
            return {
                "claims": [
                    {
                        "claim": (answer.split("。")[0] or "回答内容")[:80],
                        "label": "SUPPORTED" if score >= 4 else "NOT_IN_REFERENCE",
                        "evidence": "与给定材料比对",
                    }
                ],
                "unsupported_count": unsupported,
                "score": score,
                "reasoning": f"回答与参考材料的重合度为 {overlap:.2f}，按锚点映射到 {score} 分。",
            }

        if rubric_id == "instruction_following":
            hard = 0 if score >= 4 else (1 if score >= 3 else 2)
            return {
                "constraints": [
                    {
                        "constraint": "题面显式约束（由规则判定给出，裁判此处仅记录）",
                        "satisfied": hard == 0,
                        "evidence": "见 deterministic 层结果",
                    }
                ],
                "hard_violations": hard,
                "score": score,
                "reasoning": f"硬约束违反 {hard} 条。",
            }

        if rubric_id == "agent_trajectory":
            completed = score >= 3
            return {
                "task_completed": completed,
                "tool_sequence": list(sample.meta.get("expected_tools") or []),
                "extra_tool_calls": [] if score >= 4 else ["search_records"],
                "missing_tool_calls": [] if completed else ["aggregate"],
                "param_issues": [],
                "safety_violation": False,
                "score": score,
                "reasoning": f"任务完成={completed}，按轨迹质量映射到 {score} 分。",
            }

        # 默认走 helpfulness 的多维加权
        subs = {
            "correctness": score,
            "completeness": max(1, score - (0 if raw > 0.6 else 1)),
            "relevance": min(5, score + (0 if raw > 0.3 else -1)),
            "clarity": min(5, max(1, score)),
        }
        weights = {"correctness": 0.40, "completeness": 0.25, "relevance": 0.20, "clarity": 0.15}
        weighted = sum(subs[k] * w for k, w in weights.items())
        return {
            "restated_intent": (sample.prompt or "")[:60],
            "sub_scores": subs,
            "weighted_score": round(weighted, 2),
            "reasoning": f"子维度加权得 {weighted:.2f}。回答与参考重合度 {overlap:.2f}。",
        }

    def _judge_pairwise(self, sample: Sample, hint: dict[str, Any]) -> dict[str, Any]:
        """模拟 pairwise 裁判。

        刻意保留一点位置偏差（真难分高下时偏向先出现的那一份）。
        这样换位一致性才不是天然 100%，偏差度量那一套也才有东西可测 ——
        如果模拟出来的裁判毫无偏差，评测里关于偏差控制的部分就成了摆设。
        """
        a = hint.get("answer_a") or ""
        b = hint.get("answer_b") or ""
        ref = sample.reference or ""

        qa = self._overlap(a, ref)
        qb = self._overlap(b, ref)

        if abs(qa - qb) < 0.02:
            return {"winner": "A", "reasoning": "两者质量接近，按呈现顺序取前者。"}
        if qa > qb:
            return {"winner": "A", "reasoning": "前者与参考内容重合度更高。"}
        return {"winner": "B", "reasoning": "后者与参考内容重合度更高。"}

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        """用字符 bigram 的 F1 衡量回答与参考答案的重合度。

        刻意不用「字符集合重合率」那种写法：它对长度完全不敏感，
        长回答天然占优，模拟出来的冗长偏差会失真成 100%，
        那评测里关于偏差控制的部分就成了摆设。
        F1 同时看精确率与召回率，长度失衡会被两边共同惩罚。
        """
        if not b:
            # 没有参考答案时给一个中性偏上的估计
            return 0.55 if len(a) > 40 else 0.35
        return token_overlap_f1(a, b)
