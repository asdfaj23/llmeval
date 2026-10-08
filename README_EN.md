<div align="center">

# LLMEval · Trustworthy LLM Evaluation Framework

**Three-tier judging × bias control × judge calibration × attribution loop**

*Methodology aligned with [G-Eval](https://arxiv.org/abs/2303.16634) (rubric paradigm), [MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685) (LLM-as-a-judge, position bias), [τ-bench](https://arxiv.org/abs/2406.12045) (pass^k reliability), PoLL (judge panels), and **ByteDance Seed's evaluation system** (Seed 1.8 / 2.0 / 2.1 Model Cards)*

[简体中文](README.md) | English

[![CI](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml/badge.svg)](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-248%20passed-brightgreen.svg)](tests/)
[![Dependencies](https://img.shields.io/badge/dependencies-1%20%28PyYAML%29-orange.svg)](requirements.txt)

</div>

---

## Why this exists

The common recipe for "LLM evaluation" is: grab a few public benchmarks, run them, publish a ranking. That road has three unavoidable problems — **public benchmarks saturate fast** (vendors optimize against them, discriminative power decays within months), **they diverge from real usage** (a high leaderboard score says little about handling a messy real-world request), and **they are unexplainable** (a total score tells you nothing about where the weakness is or what data to collect next).

llmeval takes the other road: **derive evaluable categories from real use cases first**, and make evaluation *reproducible, trustworthy, and actionable*. The design aligns with the ByteDance Seed evaluation methodology (Seed1.8 three principles / Seed2.0 four-axis framework / Seed2.1 product-driven evaluation) and adds a construct-validity check inspired by the NeurIPS 2025 audit finding that only 16% of 445 benchmark papers used statistical tests when comparing models — llmeval makes paired testing the default.

## Highlights

| | |
|---|---|
| 🧱 **Three-tier judging** | Deterministic rules (full coverage, zero cost, zero bias) → LLM judges (G-Eval / pairwise / panel) → human labels (calibration only). Core trade-off: **whatever a rule can judge never goes to a judge** — ~81% of verdicts are decided by code (one full run: 1429 rule verdicts vs 344 judge verdicts) |
| 📊 **Significance testing by default** | Paired McNemar test + bootstrap 95% CIs. When a gap falls inside the noise band, the report says "insufficient evidence" instead of picking a winner |
| 🔁 **Reliability: pass^k** | An item counts only if the model passes it k times in a row (τ-bench style). The gap between pass@k and pass^k is the number deployment decisions should look at |
| ⚖️ **Judge calibration** | Cohen's κ (quadratically weighted for ordinal scores) + inter-judge agreement + median aggregation against outliers. Below κ = 0.6 the report warns: scores are advisory only, not a release gate |
| 🎭 **Bias control & disclosure** | Forced bidirectional swap in pairwise (position bias), longer-wins rate (verbosity bias), same-family judge/SUT alerting (self-preference). Bias can't be eliminated — only measured, mitigated, and **honestly disclosed** |
| 🔒 **Eval-environment isolation** | Answer-isolation self-check (catches the silent failure where the reference answer leaks into the prompt) + canary leak scanning + tool-trace audit. All deterministic, **zero API cost**. Findings deliberately stay out of capability scores — a breach is "can we trust this run", not "is the model good" |
| 🔄 **Data flywheel** | Failure attribution → data strategy suggestions → candidate item generation (gated by `needs_review`). A badcase is not the end; it is the start of the next data cycle |
| 🥊 **Agent paradigm comparison** | ReAct vs Plan-and-Execute on the same benchmark: the same model endpoint instantiated twice with different strategies, isolating *framework* differences rather than *model* differences |
| 🪶 **Minimal & reproducible** | Single runtime dependency (PyYAML); HTTP via stdlib `urllib`; statistics in pure stdlib; CPU-only; offline mock mode runs the entire pipeline with zero credentials |

## Quick start

**Requirements**: Python 3.10+, no GPU, no API key (the offline demo needs nothing).

```bash
git clone https://github.com/asdfaj23/llmeval.git
cd llmeval
pip install PyYAML        # the only runtime dependency

# 1. List suites, rubrics, capability dimensions, model status
python run.py list

# 2. Run a full evaluation (built-in mock models, fully offline)
python run.py run --suite all --pairwise

# 3. Render the report (pure local computation, no model calls)
python run.py report --run latest --open
```

You get a **self-contained single-file HTML report** at `outputs/runs/<run_id>/report.html`.

![Evaluation report preview](docs/assets/report_preview.png)

*A real evaluation report (GLM vs DeepSeek, full run: 290 items / 13 dimensions / 708 execution records; the bank has since grown to 368 items): run overview → leaderboard with 95% CIs → capability radar → dimension heatmap → significance tests, all in one self-contained HTML file.*

> **About mock**: the two enabled mock models only verify that the pipeline works. They produce simulated data and **say nothing about any real model's capability** (the report shows a red warning banner). Plug in real APIs for real results — see below.

### Connect real models

Change two files, no code:

```bash
cp .env.example .env        # fill in any provider's API key
```

```yaml
# configs/models.yaml — flip enabled to true for the models you want to test
suts:
  - id: deepseek-chat
    provider: deepseek
    model: deepseek-chat
    enabled: true             # ← here
```

Built-in providers: Volcengine Ark / DeepSeek / Moonshot / Qwen (DashScope) / Zhipu GLM / OpenAI — all OpenAI-compatible; switching providers only changes `base_url`.

## What it measures: 13 capability dimensions

Every dimension is annotated with its source of truth — real use cases or public benchmarks — **dimensions are derived from real usage, not copied from leaderboards**:

| Dimension | What it measures | Anchor |
|---|---|---|
| `knowledge` | Factuality, long-tail knowledge | Seed2.0 model card: long-tail knowledge gap |
| `instruction` | Complex instruction following | Seed2.0 model card: multi-step instruction failures |
| `reasoning` | Math, logic, multi-step reasoning | Seed principle: "push the frontier of intelligence" |
| `coding` | Code correctness, Vibe Coding | Seed2.0 four-axis framework |
| `long_context` | Extract & reason over long documents | Seed2.0 four-axis framework |
| `agent` | Tool calling, trajectories | τ-bench & GAIA paradigms |
| `multi_agent` | Division of labor, checks & balances | MultiAgentBench; Seed2.1 harness-aware evaluation |
| `knowledge_graph` | Triple extraction, multi-hop reasoning | Structured-knowledge scenarios |
| `realworld` | End-to-end task completion | Seed2.0 four-axis framework |
| `multiturn` | Multi-turn consistency | "LLMs Get Lost in Multi-Turn Conversation" |
| `multimodal` | Chart reading, counting, spatial, diagram understanding | Cross-modal data scenarios |
| `office` | Office & productivity tasks | Largest category in Seed2.1 crowdsourced tasks |
| `safety` | Refusal boundaries, hallucination, prompt injection | AI safety team scope |

## Real evaluation results

The framework has completed full 13-dimension evaluations of two production models (708 execution records, 0 call errors, temperature 0.0), plus an agent-paradigm comparison on the same benchmark:

**Model comparison (DeepSeek vs GLM)**:
- Overall 4.29 vs 4.13; paired McNemar on 241 items, p = 0.0104 — **statistically significant**
- Very different weakness profiles: the largest gap is coding (4.83 vs 3.68); knowledge-graph and office tasks split the wins
- Counter-intuitive cost: DeepSeek's total cost is ~3.3× GLM's — capability conclusions must be read together with cost
- Reliability: both models' pass^3 ≈ 0.34; unstable items 7 vs 4 — similar means, different stability

**Paradigm comparison (ReAct vs Plan-and-Execute, same endpoint)**:
- Task quality identical (both 5.0/5) — with strong models, pick a framework by task length and cost, not quality
- 3× interaction overhead difference: ReAct ~4 calls vs Plan-and-Execute ~12
- Qualitative: ReAct is lighter and adapts better to dynamic environments; Plan-and-Execute trades overhead for explicit plans, auditability, and structured failure recovery

> These numbers are snapshots and will drift as the eval set and models evolve. **How a conclusion is reached matters more than the conclusion**: every claim above carries a significance test, a confidence interval, and per-item inspectable verdict details.

## Report contents

| Section | Question it answers |
|---|---|
| Run overview | How much ran, cost, errors |
| Leaderboard | Who is stronger, **and whether the gap is significant** (bootstrap 95% CI) |
| Capability radar | Where the strengths and weaknesses are |
| Dimension matrix | Full model × dimension heatmap |
| Win-rate matrix | Pairwise relative strength (bidirectional swap; position-sensitive → tie) |
| Reliability | pass@k vs pass^k |
| Judge trustworthiness | κ — can the judge's scores be trusted at all |
| Panel agreement | Inter-judge weighted κ / exact agreement / mean divergence |
| Bias metrics | Position consistency, verbosity preference, self-preference risk — all disclosed |
| Weakness attribution & data strategy | Failure-type distribution → what data to collect next |
| Failure details | Expandable badcases with prompt/response/reference/verdict detail |

## Repository layout

```
llmeval/
├── run.py                     CLI entry point
├── requirements.txt           the only dependency: PyYAML
├── configs/                   models registry, suites, G-Eval rubrics
├── datasets/                  368 JSONL items, 13 dimensions × 3 difficulty levels
├── src/llmeval/               schema / client / mock / sut / metrics / bias /
│                              calibration / stats / pipeline / analysis / report
├── scripts/                   eval-set builder, gold-κ, human review page, badcase flywheel
├── tests/                     248 offline pytest cases (zero API calls)
└── docs/                      methodology & architecture (Chinese)
```

## Honest limitations

1. **Simulated data is not a conclusion.** Mock validates the pipeline, nothing more.
2. **No human labels → no κ.** The report leaves the trustworthiness section blank rather than substituting another metric.
3. **368 items is a seed set, not a production bank.** Items are handcrafted, not sampled from real user requests — the main gap vs industrial practice, on the roadmap.
4. **Conclusions require significance.** Inside the noise band the report says "insufficient evidence".
5. **Code execution is only weakly isolated.** Items with `tests` run in a subprocess (isolated temp dir + 10s hard timeout + static blacklist) — that is not a container sandbox. Items without `tests` fall back to the judge.

## Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) (Chinese). New dimensions, metrics, and SUTs all have standard plug-in points; the most wanted contribution is a container-level coding sandbox. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citing

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

## License

[MIT](LICENSE) © 2026 邱子策 (Zice Qiu)

## Key references

- Seed 1.8 / 2.0 / 2.1 Model Cards (Bytedance Seed) — evaluation principles, four-axis framework, product-driven evaluation
- G-Eval (Liu et al., 2023, arXiv:2303.16634) — rubric paradigm
- MT-Bench / Chatbot Arena (Zheng et al., 2023) — LLM-as-a-judge, position bias
- τ-bench (Sierra) — pass^k reliability
- Length-Controlled AlpacaEval (Dubois et al., 2024) — verbosity bias
- PoLL (Verga et al., 2024) — judge panels beat single judges
- MultiAgentBench (Zhu et al., 2025) — multi-agent evaluation
- Landis & Koch (1977) — κ interpretation bands
