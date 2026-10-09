# llmeval

An LLM evaluation framework I wrote from scratch. 368 items, 13 capability dimensions, three-tier judging, and a single self-contained HTML report at the end of one command. Runs fully offline.

[![CI](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml/badge.svg)](https://github.com/asdfaj23/llmeval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One runtime dependency (PyYAML). HTTP goes through stdlib `urllib`, the statistics are hand-rolled, CPU only, no GPU, no Docker.

中文说明见 [README.md](README.md)。

---

## Why I wrote another one

Public leaderboards are no use when you actually need to compare two models. They saturate fast (vendors optimize against them and the spread collapses within months), they drift from real usage (a high score says little about a long, messy, jumping-around request), and they only give you a number (nothing about where the weakness is or what data to collect next).

There is a quieter problem too: hardly anyone runs a significance test. The NeurIPS 2025 construct-validity audit went through 445 benchmark papers and found only 16% used statistical tests when comparing models. A 0.1 gap over ~200 items is very plausibly noise, but the report reads like a conclusion.

So I wrote this as if it were a report someone would audit. Four rules:

- If code can decide it, don't pay a judge to guess
- A gap has to survive a paired test, otherwise the report says "insufficient evidence"
- The judge gets calibrated too; below κ = 0.6 the scores are advisory
- If the environment misbehaves, the run is void. Dirty scores don't get averaged into capability

Item design and dimensions follow ByteDance Seed's evaluation practice (the three principles in Seed 1.8, the four-axis framework in 2.0, product-driven evaluation in 2.1). The point-by-point mapping is in [docs/字节评测方法对照.md](docs/字节评测方法对照.md).

## Three minutes to a run

```bash
git clone https://github.com/asdfaj23/llmeval.git
cd llmeval
pip install PyYAML                              # that's the whole dependency list

python run.py list                              # suites, rubrics, dimensions, model status
python run.py run --suite all --pairwise        # full run (mock models, offline)
python run.py report --run latest --open        # render report locally, no extra model calls
```

The report lands at `outputs/runs/<run_id>/report.html`, one file, inline SVG charts, mail it to anyone.

About mock: the two enabled mock models exist to prove the pipeline isn't broken. Their numbers are simulated, and quoting them as evidence about a real model would be wrong. The report puts a red banner at the top so you can't forget.

### Point it at your own models

Two files change, no code:

```bash
cp .env.example .env            # put your API key here
```

```yaml
# configs/models.yaml — flip enabled for whatever you want to test
suts:
  - id: deepseek-chat
    provider: deepseek
    model: deepseek-chat
    enabled: true
```

Providers built in: Volcengine Ark, DeepSeek, Moonshot, Qwen/DashScope, Zhipu, OpenAI. All OpenAI-compatible, so switching vendors is a `base_url` edit.

## The item bank: 368 items, 13 dimensions

Dimensions weren't copied off a leaderboard. They were derived from real use cases first, then made evaluable. What each dimension covers and deliberately excludes is in [docs/维度边界定义.md](docs/维度边界定义.md); items that straddle two dimensions carry `sole_judge` / `belongs_elsewhere` fields so credit is counted exactly once.

| Dimension | Items | Why it's here |
|---|---|---|
| `reasoning` | 40 | Seed's "push the frontier" principle |
| `knowledge` | 38 | Long-tail knowledge gaps, called out in the Seed 2.0 card |
| `instruction` | 32 | Complex multi-step instruction failures, same card |
| `coding` | 30 | Items with `tests` actually execute, no judge reading code |
| `safety` | 30 | Refusal boundaries, hallucination, prompt injection |
| `agent` | 28 | Tool calls and trajectories, after τ-bench / GAIA |
| `multiturn` | 28 | Information retention, from "LLMs Get Lost in Multi-Turn Conversation" |
| `multimodal` | 26 | Chart reading, counting, spatial relations; images are script-generated |
| `realworld` | 25 | End-to-end task completion |
| `knowledge_graph` | 24 | Triple extraction, multi-hop reasoning |
| `long_context` | 24 | Extract and reason over long documents |
| `multi_agent` | 24 | Division of labour and checks, after MultiAgentBench |
| `office` | 19 | The largest category in Seed 2.1's crowdsourced tasks |

Difficulty splits 49 easy / 156 medium / 163 hard. Early versions had too many easy items to separate anything, so hard got filled out to just over 40%.

## Judging happens in three tiers

```
Tier 1  Deterministic rules   full coverage, free, unbiased, reproducible
        format / length / must-contain / JSON schema / regex / MCQ / numeric tolerance / safety heuristics

Tier 2  LLM judges            sampled, costs money, needs calibration
        G-Eval single scoring / pairwise with forced swap / reference-aligned / judge panels

Tier 3  Human labels          small sample, calibrates Tier 2 rather than producing scores
        Cohen's κ (quadratically weighted for ordinal) / agreement / Spearman / MAE
```

The reasoning is mundane: whether a response stayed under a word limit is a deterministic question, and paying an LLM to answer it is both expensive and a free source of bias. Same for multi-agent failure modes, a reviewer that never rejects, a rejection nobody acts on, one role doing everything. Those are countable from the transcript.

In practice about 80% of verdicts come from code. On the full 290-item run it was 1429 rule verdicts against 344 judge verdicts.

Tier 2 carries three defences: pairwise always swaps positions (position bias), longer-wins is measured (verbosity bias), and a judge from the same family as the model under test raises an alert (self-preference). Bias can't be removed, only measured, mitigated, and disclosed. Panels aggregate on the median; PoLL convinced me of that.

## Comparing two models

Gaps only get reported if they survive a test: paired McNemar plus bootstrap 95% CIs, and when a difference sits inside the noise band the report writes "insufficient evidence to say which is stronger" instead of picking a winner.

One full real run is done, on two production models (that was the 290-item bank: 708 execution records, 0 call errors, temperature fixed at 0.0):

- Overall 4.29 vs 4.13, paired over 241 items, McNemar p = 0.0104, so the difference is real
- Weakness profiles are nothing alike: coding is the biggest gap (4.83 vs 3.68), knowledge graph and office tasks split the wins
- Cost went the unexpected way: DeepSeek ran about 3.3x GLM, so capability claims have to be read next to cost
- Reliability: both sit at pass^3 = 0.34, with 7 vs 4 unstable items. Similar means, different stability

There's also an agent-paradigm comparison on the same bank, ReAct vs Plan-and-Execute, two strategy instances over one model endpoint, so the difference is the framework and not the model. Quality tied at 5.0/5; interaction cost differed 3x (~4 calls vs ~12). With strong models you pick a framework by task length and cost, not by score.

Those numbers are a snapshot of that bank and those versions. What I care about more is how they were reached: every claim carries its test, its interval, and per-item verdict details you can expand.

## The part I added last: eval-environment isolation

An interviewer asked whether an agent driven over API should be sandboxed so it can't go looking for answers. Fair question, and the failure mode is nasty: the reference answer sits right there, the model copies it, scores full marks, and nothing in the report looks wrong. No crash, no error, just a wasted run that reads like a triumph.

`src/llmeval/containment.py` runs three audits, all deterministic and free:

- Answer isolation: protected fragments (references, gold labels, expected test values) must not appear in anything the model can see. Fragments under 12 characters are skipped, because an MCQ gold of "B" showing up in the item is normal, and a check that cries wolf gets ignored
- Canary scan: each item derives a deterministic marker from its id. If that marker shows up in a response or a tool trace, the model reached something it shouldn't have. The scan looks for every marker, not just its own, since another item's marker means cross-contamination
- Tool trajectory audit: calls outside the whitelist, path traversal, system directories, answer filenames, external hosts. All logged, all reviewable. The blacklist is deliberately narrow; words like `format` or `del` don't belong in it

Full-bank result: 368 items, 0 leaks. The check runs in CI, so anyone who writes an answer into the prompt gets blocked at the pull request. A companion script (`scripts/audit_reference.py`) checks reference answers against their own constraints, and it caught a real bug on the spot: one instruction item's reference answer exceeded the word limit stated in its own prompt, so full marks were unreachable and every verdict on it was wrong.

To be clear about the ceiling: this is weak isolation, not a container sandbox. No Docker, no seccomp, no forced network cut. Every tool here is a pure function reading `context`, writing nothing and calling nowhere, so at this stage there isn't much to isolate. Isolation findings also stay out of capability scores on purpose: a breach answers "can this run be trusted", not "is the model good", and averaging it into capability dilutes exactly the signal that should trigger a rerun. Cost-ordered eight-layer plan in [docs/沙盒与隔离.md](docs/沙盒与隔离.md).

## What the report looks like

![Evaluation report](docs/assets/report_preview.png)

That's the real report from the GLM vs DeepSeek run: overview, leaderboard with 95% CIs, capability radar, dimension heatmap, significance tests, all in one HTML file.

Beyond scores it reports judge trustworthiness (κ below threshold gets marked advisory, missing human labels leaves the cell blank rather than substituting another metric), position consistency, verbosity preference, failure-type distribution mapped to what data to collect next, and expandable badcases. Mock runs always show the red banner.

## Layout

```
llmeval/
├── run.py                 CLI entry
├── configs/
│   ├── models.yaml        SUTs, judges, runtime params
│   ├── suites/            suite = datasets × metrics × models
│   └── rubrics/           G-Eval style rubrics (criterion + steps + anchors)
├── datasets/              368 JSONL items, schema.md for writing more
│   └── calibration/       human labels for judge calibration
├── src/llmeval/
│   ├── schema.py          data structures + dimension table
│   ├── client.py          one HTTP client: caching, retries, concurrency
│   ├── sut/               chat / agent / multi_agent / multi_turn
│   ├── metrics/           the three tiers live here
│   ├── containment.py     isolation audits
│   ├── bias.py            bias metrics
│   ├── calibration.py     κ and panel agreement
│   ├── stats.py           McNemar + bootstrap
│   ├── pipeline.py        orchestration
│   ├── analysis.py        aggregation, attribution, data strategy
│   └── report.py          single-file HTML report
├── scripts/               bank builder, gold κ, blind review page, badcase loop
├── tests/                 256 offline pytest cases, zero API calls
└── docs/                  methodology, architecture, commands, isolation
```

## Commands

```bash
python run.py list                                       # suites / rubrics / dimensions / models
python run.py run --suite all --pairwise                 # full run with pairwise
python run.py run --suite agent                          # agent only (repeats=3, for pass^k)
python run.py run --suite all --limit 20                 # 20 items first, don't burn budget
python run.py report --run latest --open                 # render, costs nothing
python run.py calibrate --run latest --gold <gold.jsonl> # calibrate judges against human labels
python run.py failures --run latest --top 5              # quick badcase look in the terminal
python scripts/audit_isolation.py                        # full-bank answer isolation check
```

## What it doesn't do yet

Worth reading before the feature list:

1. Mock output is not a result. It proves the pipeline runs.
2. No human labels means no κ, and that cell stays empty in the report.
3. 368 handcrafted items is a seed set, not a production bank. Nothing here is sampled from real user requests, which is the biggest gap versus industrial evaluation and the first thing on the roadmap.
4. Code execution is subprocess-level weak isolation (own temp dir, 10s hard timeout, static blacklist), not a container sandbox. Items without `tests` fall back to the judge.
5. Isolation is post-hoc audit only; there is no in-flight interception.

Roadmap: container-level coding sandbox, repeats across all dimensions so pass^k covers everything, real user request sampling, English item bank and internationalized reports.

## Docs

| Doc | Contents |
|---|---|
| [评测体系](docs/评测体系.md) | Where dimensions come from, rubric design, metric definitions, the SOP |
| [架构说明](docs/架构说明.md) | Code structure, data flow, how to add a dimension / metric / SUT |
| [命令手册](docs/命令手册.md) | Every command plus six typical workflows |
| [字节评测方法对照](docs/字节评测方法对照.md) | Each design mapped to Seed method and paper |
| [沙盒与隔离](docs/沙盒与隔离.md) | How far isolation goes, why not containers yet |
| [维度边界定义](docs/维度边界定义.md) | In/out-of-scope per dimension, rules for ambiguous items |
| [指标与维度对照](docs/指标与维度对照.md) | Public benchmarks, metrics, tier and pass line per dimension |
| [datasets/schema.md](datasets/schema.md) | Read this before contributing items |

## Contributing

Issues and PRs welcome; integration points are described in [CONTRIBUTING.md](CONTRIBUTING.md) (Chinese). New dimensions, metrics and SUTs all have standard hooks, and the contribution I most want is a container-level coding sandbox. If you add items, read `datasets/schema.md` first; CI runs the isolation check.

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

## References

- Seed 1.8 / 2.0 / 2.1 Model Cards (ByteDance Seed)
- G-Eval (Liu et al., 2023, arXiv:2303.16634)
- MT-Bench / Chatbot Arena (Zheng et al., 2023)
- τ-bench (Sierra)
- Length-Controlled AlpacaEval (Dubois et al., 2024)
- PoLL (Verga et al., 2024)
- MultiAgentBench (Zhu et al., 2025)
- Landis & Koch (1977)

## License

[MIT](LICENSE) © 2026 邱子策 (Zice Qiu)
