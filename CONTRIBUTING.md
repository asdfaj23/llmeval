# 贡献指南

感谢你对 llmeval 的关注！这个项目的核心主张是「评测结论必须可信」，贡献请围绕这一点展开。

## 开发环境

```bash
git clone https://github.com/<你的用户名>/llmeval.git
cd llmeval
pip install PyYAML pytest   # 运行时唯一依赖 PyYAML，pytest 仅开发用
```

不需要 GPU，不需要 API key：全部测试离线运行，README 的「30 秒跑起来」用内置 mock 即可完成。

## 提交前检查

- [ ] `python -m pytest tests -q` 全绿
- [ ] 测试保持**离线**：不调任何真实 API、不依赖网络。依赖外部服务的测试会变成时好时坏的东西，最后没人跑
- [ ] 不引入新的运行时依赖（例外需要充分理由并先开 Issue 讨论）
- [ ] 新增判定逻辑时有对应测试
- [ ] 文档同步更新（`docs/评测体系.md` / `docs/架构说明.md`）

## 什么贡献最被欢迎

按优先级排序：

1. **代码执行沙箱**：`coding` 维度目前靠裁判读代码打分，接一个子进程沙箱执行测试用例是最有价值的扩展（见 `docs/架构说明.md` § 七）
2. **新的确定性指标**：`metrics/deterministic.py` 的函数签名固定，返回 `None` 表示不适用
3. **新的评测维度**：在 `schema.py` 的 `DIMENSIONS` 里加条目时**必须写清对齐依据**——这是这张表存在的意义
4. **新的被测对象（SUT）**：继承 `BaseSUT` 实现一个方法即可，判定层/报告层不用改
5. **评分表（rubric）改进**：改 `configs/rubrics/*.yaml`。注意：评分表一变，旧 κ 全部作废，必须重新校准
6. **文档与翻译**：英文文档、使用案例、评测方法学文章都很欢迎

## 提交规范

- Commit message 用祈使句：`add pass^k plot to report`，不用 `added xxx`
- 一个 PR 只做一件事
- 涉及评测方法学的改动（指标口径、校准流程），请在 PR 描述里写清依据（论文 / 模型卡 / 数据）
