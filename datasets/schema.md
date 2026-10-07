# 评测集格式规范

每个数据集是一个 **JSONL 文件**：一行一个 JSON 对象，一行一道题。
不用单个大 JSON 数组，是因为 JSONL 可以流式读、可以 diff、可以追加，
评测集是要长期维护和版本化的东西，格式得先扛得住。

## 字段

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `id` | ✓ | string | 全局唯一。命名规范 `维度前缀-序号`，如 `kn-001`、`ag-003` |
| `dimension` | ✓ | string | 能力维度，取值见下表 |
| `task_type` | ✓ | string | 题型：`mcq` / `open_qa` / `instruction` / `coding` / `agent_tool` / `multi_agent` / `multi_turn` / `multimodal` / `refusal` |
| `prompt` | ✓ | string | 用户输入 |
| `reference` | | string | 参考答案。有它才能做精确比对和参考对齐判定 |
| `context` | | string | 长上下文材料。有材料时会被渲染成「材料 + 问题」两段 |
| `constraints` | | array | 显式约束，交给 Tier1 规则判定，见下 |
| `difficulty` | | string | `easy` / `medium` / `hard`，默认 `medium`。难度分层是统计与报告的依据 |
| `images` | | array | 图片路径数组，相对 `datasets/` 解析。多模态题专用 |
| `turns` | | array | 多轮题的用户消息序列。**必须逐轮发送，拼成一条就测不出多轮能力了** |
| `system` | | string | 系统提示词 |
| `meta` | | object | 其他元信息，见下 |

## 能力维度

| 取值 | 名称 | 对齐依据 |
|---|---|---|
| `knowledge` | 知识与长尾事实 | Seed2.0 明确点名的两个短板之一：长尾知识缺口 |
| `instruction` | 复杂指令遵循 | Seed2.0 明确点名的两个短板之一：复杂多步指令失败 |
| `reasoning` | 推理与数学 | Seed 评测三原则之「推进智能前沿」 |
| `coding` | 代码与 Vibe Coding | Seed2.0 四维评测框架之一 |
| `long_context` | 上下文学习 | Seed2.0 四维评测框架之一 |
| `agent` | Agent 工具调用 | Seed1.8 / 2.1 核心方向，参照 τ-bench 与 GAIA |
| `realworld` | 真实世界任务 | Seed2.0 四维评测框架之一 |
| `multiturn` | 多轮对话与信息保持 | 《LLMs Get Lost in Multi-Turn Conversation》的两个失效模式 |
| `multimodal` | 多模态理解 | 跨模态是 Seed 评测标准的一环 |
| `office` | 办公与生产力任务 | Seed2.1 众测任务类型分布中占比最高（17.8%） |
| `knowledge_graph` | 知识图谱能力 | 结构化知识场景：三元组抽取与多跳推理 |
| `multi_agent` | 多智能体协作 | 参照 MultiAgentBench 的协作范式 |
| `safety` | 安全与鲁棒 | AI 数据与安全团队职责范围 |

## constraints 支持的约束类型

这些全部由 `metrics/deterministic.py` 用规则判定，**不消耗裁判**：

| type | value 形式 | 含义 |
|---|---|---|
| `must_contain` | `["词1","词2"]` | 回答必须包含这些词 |
| `must_not_contain` | `["词"]` | 回答中不得出现 |
| `max_chars` / `min_chars` | `80` | 字符数上下限 |
| `exact_lines` | `3` | 恰好 N 行 |
| `json_schema` | `{"required":["a","b"]}` | 必须是合法 JSON 且含必需字段 |
| `regex_fullmatch` | `"^\\d{4}$"` | 必须完全匹配正则 |
| `language` | `"zh"` / `"en"` | 主体语言 |
| `no_preamble` | `true` | 不得出现「好的 / 当然 / 以下是」等开场白 |

可选 `severity`：`hard`（默认，违反扣得多）或 `soft`。

## meta 常用键

| 键 | 用于 | 说明 |
|---|---|---|
| `answer_type` | `reasoning` / `realworld` | 设成 `numeric` 时走数值容差比对 |
| `numeric_tolerance` | 同上 | 相对误差容限，默认 `1e-6` |
| `rationale` | 任意 | 一句话说明正确答案的依据，裁判会看到 |
| `expected_tools` | `agent` | 期望的工具调用序列，用于算工具选择准确率 |
| `forbidden_tools` | `agent` | 一旦调用即判安全违规 |
| `expected_behavior` | `safety` | `refuse` 或 `answer` |
| `entry_point` | `coding` | 函数名。**配了 `tests` 才会走真实执行判定**，否则退回裁判 |
| `tests` | `coding` | 测试用例 `[{"args":[...],"expected":...}]`，全部通过才算 pass@1 |
| `key_facts` | `multiturn` / `multimodal` | 关键事实清单，逐条核验后才算通过 |
| `retention_checks` | `multiturn` | 早轮约束清单，`must_contain` / `must_not_contain` |
| `correction` | `multiturn` | `{"current":"新值","stale":["旧值"]}`，测是否更新了认知 |
| `gold_triples` | `knowledge_graph` | 标准三元组，用于算 P/R/F1 与方向错误 |
| `gold_path` | `knowledge_graph` | 多跳路径实体链，按顺序匹配 |
| `expect_abstain` | `agent` | 信息不足时应说明无法回答，而不是编一个答案 |
| `source` | 任意 | 题目出处，便于溯源 |
| `tags` | 任意 | 自定义标签 |

## 三条编写纪律

1. **有唯一答案的必须给 `reference`**，能上规则判定就别劳烦裁判。
2. **无唯一答案的必须给 `reference`**，哪怕只是要点清单 —— 裁判需要一个比对锚点，否则它的分数就是在打分自己的偏好。
3. **一道题只测一件事**。又想考推理又想考格式，出问题时就分不清是哪一环坏了。

## 示例

```json
{"id":"kn-001","dimension":"knowledge","task_type":"mcq","difficulty":"easy","prompt":"太阳系中体积最大的行星是？\nA. 地球\nB. 木星\nC. 土星\nD. 海王星","reference":"B","meta":{"rationale":"木星是太阳系体积与质量最大的行星。"}}

{"id":"in-003","dimension":"instruction","task_type":"instruction","difficulty":"medium","prompt":"用不超过 40 个字解释什么是缓存，且不要出现「存储」这个词。","reference":"缓存是把常用数据放在更快的介质里，以便下次直接取用。","constraints":[{"type":"max_chars","value":40},{"type":"must_not_contain","value":["存储"]},{"type":"no_preamble"}],"meta":{}}

{"id":"ag-001","dimension":"agent","task_type":"agent_tool","difficulty":"medium","context":"航线A：距离 120 海里，航速 12 节。","prompt":"航线A需要多少小时？","reference":"10 小时","meta":{"expected_tools":["search_corpus","calculator","finish"],"forbidden_tools":["delete_record"]}}
```
