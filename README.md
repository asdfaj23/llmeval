# llmeval

自己写的一套大模型评测框架。368 道题、13 个能力维度、三层判定，跑完一条命令出一份单文件 HTML 报告，全程离线也能跑。

[![CI](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml/badge.svg)](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

运行时依赖只有一个 PyYAML：HTTP 走标准库 `urllib`，统计也是手写的，纯 CPU 可跑，不要 GPU，也不要 Docker。

英文说明在 [README_EN.md](README_EN.md)。

---

## 为什么要自己写一个

要给两个模型做横向对比，公开榜单指望不上，三个老毛病大家都熟：饱和太快，厂商针对性优化一轮，区分度几个月就磨没了；跟真实使用脱节，榜上高分不代表能应付用户那种又长又跳的请求；只有总分，差在哪、下一步该补什么数据，它一个字都不说。

还有个更隐蔽的问题：比较少数的模型时基本没人做检验。NeurIPS 2025 那篇构念效度审计翻了 445 篇 benchmark 论文，在比较模型时用了统计检验的只有 16%。两百多题上 0.1 的分差很可能是噪声，但报告读起来像个结论。

所以我干脆把这个项目当成"一份要给人复查的报告"来写，立了四条规矩：

- 能用代码判的，不交给 LLM 裁判
- 分差必须过配对检验，过不了就写"不足以判断谁强"
- 裁判自己也要被校准，κ 不到 0.6 的分数只当参考
- 环境出了问题就作废这一轮，不把脏分数平均进能力分

题目和维度参照字节 Seed 的评测体系（Seed 1.8 / 2.0 / 2.1 模型卡里的三原则、四维框架、产品驱动评测），逐条对照写在 [docs/字节评测方法对照.md](docs/字节评测方法对照.md)。

## 三分钟跑起来

```bash
git clone https://github.com/asdfaj23/llmeval.git
cd llmeval
pip install PyYAML                              # 就这一个依赖

python run.py list                              # 看看有哪些套件、评分表、维度、模型
python run.py run --suite all --pairwise        # 跑一轮完整评测（默认 mock 模型，离线）
python run.py report --run latest --open        # 出报告，纯本地算，不再调模型
```

报告在 `outputs/runs/<run_id>/report.html`，单文件自包含，图表是内联 SVG，发给别人直接打开。

关于 mock：默认开着的两个模拟模型只是为了验证流水线本身没坏，产出的是模拟数据，拿它们说任何真实模型的能力都是错的。报告页首会挂一条红色警示，提醒你这件事。

### 接自己的模型

改两个文件，代码一行不用动：

```bash
cp .env.example .env            # 填 API key
```

```yaml
# configs/models.yaml，把要测的模型 enabled 打开
suts:
  - id: deepseek-chat
    provider: deepseek
    model: deepseek-chat
    enabled: true
```

provider 内置了火山方舟、DeepSeek、Moonshot、通义、智谱、OpenAI，全部走 OpenAI 兼容协议，换服务商只改 `base_url`。

## 题库：368 题，13 个维度

维度不是从公开榜单抄的，是先梳理真实用例、再倒推出来的。每个维度管什么、不管什么，都写在 [docs/维度边界定义.md](docs/维度边界定义.md) 里，容易混的题（比如办公任务里也带推理）有 `sole_judge` / `belongs_elsewhere` 两个字段做归属判定，一题只记一次分。

| 维度 | 题数 | 为什么有这一维 |
|---|---|---|
| `reasoning` | 40 | Seed 三原则里的"推进智能前沿" |
| `knowledge` | 38 | Seed 2.0 模型卡点名的长尾知识缺口 |
| `instruction` | 32 | 复杂多步指令失败，同样是模型卡里的短板 |
| `coding` | 30 | 配了 `tests` 的题走真实执行，不靠裁判读代码 |
| `safety` | 30 | 拒答边界、幻觉、提示注入 |
| `agent` | 28 | 工具调用与轨迹，参照 τ-bench / GAIA |
| `multiturn` | 28 | 多轮信息保持，出自《LLMs Get Lost in Multi-Turn Conversation》 |
| `multimodal` | 26 | 图表读数、计数、空间关系，配图是脚本生成的 |
| `realworld` | 25 | 端到端任务完成 |
| `knowledge_graph` | 24 | 三元组抽取与多跳推理 |
| `long_context` | 24 | 长文档里提取并推理 |
| `multi_agent` | 24 | 分工与制衡，参照 MultiAgentBench |
| `office` | 19 | Seed 2.1 众测任务里占比最高的一类 |

难度分布是 easy 49 / medium 156 / hard 163。早期版本简单题太多，拉不开差距，后来把 hard 补到了四成多。

## 打分分三层

```
第一层  确定性规则    全量跑，零成本，零偏差，可复现
        格式 / 字数 / 必含禁含 / JSON schema / 正则 / 选择题 / 数值容差 / 安全启发式

第二层  LLM 裁判      抽样跑，有成本，需要校准
        G-Eval 单点打分 / pairwise 双向换位 / 参考对齐 / 多裁判评审团

第三层  人工标注      小样本，只用来校准第二层，不直接拿来打分
        Cohen's κ（有序分用二次加权）/ 一致率 / Spearman / MAE
```

这么分的道理很直白：字数有没有超，是个确定性问题，让 LLM 去判既贵又不稳，还白白引入偏差。多智能体的那些失效模式也一样，审查者从不否决、否决了没人改、某个角色包办一切，这些在消息记录上直接数得出来，没必要交给裁判猜。

实际跑下来，大约八成判定是代码给的。那次 290 题的全量评测里，规则判定 1429 条，LLM 裁判 344 条。

裁判那一层做了三件防身的事：pairwise 强制双向换位（位置偏差）、统计冗长获胜率（冗长偏差）、裁判和被测同族时自动告警（自偏好）。偏差消不掉，只能度量、缓解，然后在报告里写清楚。多个裁判组成评审团时取中位数聚合，PoLL 那篇的结论我信。

## 比较两个模型

分差一定要过检验才敢写。用的是配对 McNemar 加 bootstrap 95% 置信区间，落在噪声区间里就直接写"不足以判断谁更强"，不挑赢家。

已经在两款主流模型上跑过一次全量（290 题那一版题库，708 条执行记录，0 调用错误，温度固定 0.0）：

- 综合 4.29 对 4.13，配对 241 题，McNemar p = 0.0104，这个差异是显著的
- 短板画像完全不同：代码差距最大（4.83 对 3.68），知识图谱和办公任务互有胜负
- 成本反直觉：DeepSeek 的综合花费约是 GLM 的 3.3 倍，能力结论得连着成本一起读
- 可靠性：两边 pass^3 都是 0.34，不稳定题 7 对 4，平均分接近，稳定性不一样

同一基准上还做了 Agent 范式对比，ReAct 对 Plan-and-Execute，同一个模型端点开两个 strategy 实例，这样比出来的是框架差别而不是模型差别：任务质量打成平手（都是 5.0/5），交互开销差了三倍（约 4 轮对约 12 轮）。强模型底下选框架，看的是任务长度和成本，不是分数。

这些数字只是当时的快照，题库和模型都会变。比起结论本身，我更在意结论是怎么来的：每一条都挂着检验、置信区间，以及逐条可展开的判定依据。

## 顺手补的一环：评测环境隔离

面试官问了个问题我觉得在理：你用 API 调智能体做题，要不要把它关进沙盒，防止它翻答案。

这事确实要命。评测有一种静默失效，答案就摆在模型眼前，它照抄一遍拿满分，报告上不报错也不崩溃，只显示"这个模型真强"，整轮结果就这么废了。

`src/llmeval/containment.py` 里做了三项审计，全部确定性、零 API 成本：

- 答案隔离自检：参考答案、金标、测试期望值这些受保护片段，不允许出现在模型能看到的材料里。小于 12 字的片段不比，选择题的 gold「B」出现在题面很正常，一个总在误报的检查最后没人会看
- canary 泄漏扫描：每题一个由 sample id 派生的确定性标记，出现在回答或工具轨迹里就说明它够到了不该够的东西。扫的是所有标记，不只是这道题自己的
- 工具轨迹审计：白名单外的工具调用、路径穿越、系统目录、答案文件名、外网地址，全都留痕可查。黑名单刻意收得很窄，`format`、`del` 这种词进不了名单

全库自检的结果是 368 题零泄漏，这条检查也进了 CI，以后谁把答案写进题面，CI 直接拦。同一批还加了个参考答案自洽性检查（`scripts/audit_reference.py`），当场抓到一道真 bug：指令题的参考答案自己就超了题面写的字数上限，那种题满分根本拿不到，判多少分都是错的。

要说清楚的是：这是弱隔离，不是容器级沙箱，没有 Docker、没有 seccomp、没有强制断网。项目里的工具都是纯函数，只读 `context`，不写盘不联网，现阶段真没多少东西可隔。隔离结论也不计入质量分，越界说明的是"这轮结果还能不能用"，不是"模型好不好"，混进能力分只会把该作废重跑的信号稀释掉。细节和八层防护的成本排序在 [docs/沙盒与隔离.md](docs/沙盒与隔离.md)。

## 报告长什么样

![评测报告预览](docs/assets/report_preview.png)

上面是那次 GLM 对 DeepSeek 全量跑出来的真实报告：运行概览、模型榜单（带 95% 置信区间）、能力雷达、维度热力表、显著性检验，都在一个 HTML 文件里。

除了分数，报告还会说这几件事：裁判可信度（κ 不够就标"仅供参考"，缺人工标注就留空，不拿别的指标顶替）、位置一致性和冗长偏好、失败类型分布对应该补什么数据、以及逐条可展开的 badcase。用 mock 跑的时候页首必有红条。

## 目录结构

```
llmeval/
├── run.py                 命令入口
├── configs/
│   ├── models.yaml        被测 / 裁判 / 运行时参数
│   ├── suites/            套件 = 数据集 × 指标 × 模型
│   └── rubrics/           评分表（criterion + steps + anchors）
├── datasets/              368 题 JSONL，另有 schema.md 写出题规范
│   └── calibration/       人工标注样例，给裁判校准用
├── src/llmeval/
│   ├── schema.py          数据结构 + 维度表
│   ├── client.py          统一客户端（缓存 / 重试 / 并发）
│   ├── sut/               被测：chat / agent / multi_agent / multi_turn
│   ├── metrics/           三层判定都在这儿
│   ├── containment.py     隔离审计
│   ├── bias.py            偏差度量
│   ├── calibration.py     κ 与评审团一致性
│   ├── stats.py           McNemar + bootstrap
│   ├── pipeline.py        流水线编排
│   ├── analysis.py        聚合、归因、数据策略
│   └── report.py          单文件 HTML 报告
├── scripts/               建题库 / 金标 κ / 人工盲评页 / badcase 飞轮
├── tests/                 256 条离线单测，一个 API 都不调
└── docs/                  方法、架构、命令、隔离
```

## 常用命令

```bash
python run.py list                                       # 套件 / 评分表 / 维度 / 模型状态
python run.py run --suite all --pairwise                 # 全量评测，带两两对比
python run.py run --suite agent                          # 只跑 Agent（repeats=3，算 pass^k）
python run.py run --suite all --limit 20                 # 先跑 20 题试试水，别一上来就烧钱
python run.py report --run latest --open                 # 出报告，不花钱
python run.py calibrate --run latest --gold <gold.jsonl> # 拿人工标注校准裁判
python run.py failures --run latest --top 5              # 终端里快速看几条失败样本
python scripts/audit_isolation.py                        # 全库答案隔离自检，离线
```

## 它现在还差什么

丑话说在前面，这几条比功能清单重要：

1. mock 只验证流水线，不是结论，别拿它的分数说话。
2. 没有人工标注就算不出 κ，报告的裁判可信度那一栏会空着。
3. 368 题是我自己构造的种子集，不是从真实用户请求里采的，这是跟工业评测最大的差距，也是接下来最想补的。
4. 代码执行只是子进程弱隔离（独立临时目录 + 10 秒超时 + 静态黑名单），不是容器沙箱；没配 `tests` 的题退回裁判。
5. 隔离只有事后审计，没有事中拦截。

路线图：容器级 coding 沙箱、全维度 repeats 让 pass^k 覆盖所有维度、真实用户请求采样、英文题库与报告国际化。

## 文档

| 文档 | 讲什么 |
|---|---|
| [评测体系](docs/评测体系.md) | 维度怎么来的、评分表怎么设计、指标口径、评估 SOP |
| [架构说明](docs/架构说明.md) | 代码结构和数据流，怎么加维度、指标、被测对象 |
| [命令手册](docs/命令手册.md) | 全部命令和六个典型场景 |
| [字节评测方法对照](docs/字节评测方法对照.md) | 每条设计对应的 Seed 方法与论文出处 |
| [沙盒与隔离](docs/沙盒与隔离.md) | 隔离做到哪一层，为什么不做容器 |
| [维度边界定义](docs/维度边界定义.md) | 每个维度管什么不管什么，易混题怎么判 |
| [指标与维度对照](docs/指标与维度对照.md) | 各维度对应的公开 benchmark、指标、判定层级与通过线 |
| [评测集编写规范](datasets/schema.md) | 想加题先读这个 |

## 贡献

Issue 和 PR 都欢迎，接入方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。新维度、新指标、新 SUT 都有标准挂载点，最想要有人补的是容器级 coding 沙箱。加题请先读 `datasets/schema.md`，CI 会跑隔离自检。

## 引用

```bibtex
@software{qiu2026llmeval,
  author  = {Qiu, Zice},
  title   = {llmeval: A Pluggable LLM Evaluation Framework with
             Three-Tier Judging, Bias Control and Judge Calibration},
  year    = {2026},
  url     = {https://github.com/asdfaj23/llmeval},
  license = {MIT}
}
```

## 主要参考

- Seed 1.8 / 2.0 / 2.1 Model Card（ByteDance Seed）：评测三原则、四维框架、产品驱动评测
- G-Eval（Liu et al., 2023, arXiv:2303.16634）：评分表范式
- MT-Bench / Chatbot Arena（Zheng et al., 2023）：LLM-as-a-Judge 与位置偏差
- τ-bench（Sierra）：pass^k 可靠性
- Length-Controlled AlpacaEval（Dubois et al., 2024）：冗长偏差
- PoLL（Verga et al., 2024）：裁判委员会优于单一裁判
- MultiAgentBench（Zhu et al., 2025）：多智能体评测
- Landis & Koch (1977)：κ 判读区间

## License

[MIT](LICENSE) © 2026 邱子策 (Zice Qiu)
