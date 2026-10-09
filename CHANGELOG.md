# 变更日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- 评测环境隔离：`src/llmeval/containment.py` 提供答案隔离自检、canary 泄漏扫描、工具轨迹审计，全部确定性、零 API 成本。判定串为 `decisive=False`，只披露不进质量分
- `scripts/audit_isolation.py`（答案隔离）与 `scripts/audit_reference.py`（参考答案自洽性）接入 CI，前者支持 `--fail-on-leak`
- `docs/沙盒与隔离.md`、`docs/维度边界定义.md`
- 自建新增 74 题，题库 294 → 368，hard 占比 31.3% → 44.3%（easy 49 / medium 156 / hard 163）
- 离线单测 217 → 256 条

### 变更

- 维度边界重构：13 个维度区分「能力轴 / 形态轴」，每个维度补 `sole_judge` 与 `belongs_elsewhere`，并定义四步归属流程；按该规则重归类 3 道边界错位题（office → reasoning / realworld）
- 修复三个既有缺陷：3 处协作诊断量漏标 `decisive=False`（把 multi_agent 维度从 4 分抬到 57.7）；in-011 的参考答案超出题面自定字数上限，满分不可达；`json_schema` 不支持数组形态
- README 与 README_EN 重写为作者口吻，去掉标语式排版，数字与题库现状对齐

### 计划中

- `coding` 维度升级到容器级沙箱（当前为子进程弱隔离：独立临时目录 + 10 秒超时 + 静态黑名单）
- 除 agent 之外的其他维度开启 `repeats`，使 pass^k 覆盖全部维度
- 评测集从 368 题种子规模继续扩充，逐步引入真实用户请求采样

## [0.1.0] - 2026-10-07

首个公开版本。

### 核心能力

- **三层判定架构**：确定性规则（全量、零成本、可复现）→ LLM 裁判（G-Eval / pairwise / 参考对齐，支持多裁判评审团 + 中位数聚合）→ 人工标注（只用于校准，报告 Cohen's κ）
- **13 个能力维度**：知识、指令遵循、推理、代码、长上下文、知识图谱、Agent 工具调用、多智能体协作、真实世界任务、多轮对话、多模态、办公任务、安全；每个维度标注对齐依据（Seed 评测体系 / 公开基准）
- **统计显著性**：配对 McNemar 检验 + bootstrap 95% 置信区间，分差不显著时如实报告「不足以判断」
- **可靠性指标**：pass@k 与 pass^k（τ-bench 范式），agent 维度 repeats=3
- **裁判校准**：加权 κ、裁判 vs 人工一致率、评审团内部一致性、κ<0.6 自动挂警示
- **偏差控制**：pairwise 双向换位（位置偏差）、冗长获胜率度量（冗长偏差）、裁判与被测同族告警（自偏好）
- **数据飞轮**：失败归因 → 数据策略建议 → 候选题生成（带 `needs_review` 门禁）的闭环脚本
- **范式对比**：ReAct vs Plan-and-Execute，同一模型端点双 strategy 实例化，隔离框架差异
- **单文件 HTML 报告**：榜单、雷达图、热力表、胜率矩阵、偏差披露、badcase 明细，纯本地渲染
- **极简依赖**：运行时仅 PyYAML；HTTP 走标准库 urllib；纯 CPU 可跑；内置 mock 离线跑通全流程

### 附带资产

- 290+ 道自建评测题（JSONL，含参考答案与显式约束），覆盖 13 维度、易中难三层
- 6 张 G-Eval 形态评分表（criterion + steps + anchors + JSON 契约）
- 全离线 pytest 测试套件（217 条）
- 中文方法学文档：评测体系设计、架构说明、命令手册、字节 Seed 方法对照

[Unreleased]: https://github.com/asdfaj23/llmeval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/asdfaj23/llmeval/releases/tag/v0.1.0
