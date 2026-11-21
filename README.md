# rsi-agent — 一个真能跑、会自我改进、且不自证的 RSI agent

对上一轮 RSI 审计结论的直接实现：**P1 真证据（方法学干净）、P2 循环论证（标签由特征加噪构造，AUC≈0.997 是同义反复）、P3 只是合成演示**。这个 agent 把三条教训变成六道可执行机制，并在一个 CPU 秒级任务上跑出可被外部确认的真实提升。

## 一句话结论

`python3 run_demo.py --generations 60` → `verdict=SELF_IMPROVEMENT_VERIFIED`，权重从 `(1, 0, 0)` 收敛到 `≈(0.68, −0.49, 0.89)`（隐藏真值 `w*=(0.9, −0.6, 1.2)` 的方向），公开分 `0.751 → 1.245`，审计链 240 条自校验 OK，**41 个反证/负对照测试全过**（含 NC 前置闸、ALE-Bench / SWE-bench 适配器冒烟）。

## 六道反自证机制

| 机制 | 对应教训 | 实现 | 反证测试 |
|---|---|---|---|
| 密封私集确认闸 | P2 循环论证 | `barriers.py`：public/private 用不同 seed；InnerAgent 只接受 `PublicHistory`（类型断言拒绝一切其它输入）；私集只在公开闸过后方可求值一次/代 | T1 |
| 非循环评分 | P2 同义反复 | `tasks.py`：reward = 被选 action 的特征 · **隐藏 w\*** + 确定性噪声；w\* 从不出现在特征里，策略只靠公开反馈搜索；外部 `ScoreVector` 由 evaluator 算，策略代码无评分入口 | T2 |
| 配对 + ESS + effect + Holm 闸 | P1 的 ESS 教训 | `gates.py`：逐实例配对差（去实例难度方差）、lag-1 自相关修正 ESS、最小实质提升比例、Holm-Bonferroni 多重比较校正；低 ESS / 低于噪声 / 不显著全部 fail-closed | T3/T4/T5 |
| 哈希链审计 | P3 合成演示 | `audit.py`：每条 `{seq,prev_hash,kind,payload,hash}`，`verify()` 可离线重放；改内容、删记录、重排都能被抓到 | T6 |
| 逐实例回归负担闸 | C 线 Artifact-to-Capability 属性 4 | `gates.py::paired_gate(max_regression_burden=0.25)`：均值过闸但少数实例被压垮（负担 >25%）即拒；真提升的低负担不误杀 | 回归负担测试 |
| NC 前置 model-free 审计 | B 线 NC 威胁最大 | `nc_guard.py`：分数进 outer 闸前，在 probe grid 上量 selection_error / signed_gap / saturation；>0.15 / 乐观 / 饱和即 fail-closed，判定进审计链 | NC 10 项 |

## 诚实判决（只有四种）

- `SELF_IMPROVEMENT_VERIFIED`：公开闸与密封确认闸都过 + 审计链完整
- `NO_VERIFIED_GAIN`：搜索过但没有一道确认闸过（"没测出提升"是合法结论，不编造）
- `AUDIT_FAILURE`：审计链自校验失败，fail-closed
- `NC_AUDIT_FAILED`：`--nc-probes > 0` 时 scorer 未过 NC 前置审计，主循环一步未跑即拒（sealed 零求值）

绝不出现"因为分数高就说成功"——分数只对外部 scorer 负责，且必须过密封集。

## 运行

```bash
cd rsi-agent
python3 run_demo.py --self-test                       # 41 个反证/负对照（exit 0 = 全过）
python3 run_demo.py --generations 60 --out results/demo.json   # 完整闭环
python3 run_demo.py --generations 60 --nc-probes 60   # 带 NC 前置审计（诚实 scorer 应通过）
python3 run_demo.py --generations 1                    # 看 NO_VERIFIED_GAIN 的诚实判决
python3 scripts/check_ale_readiness.py                # ALE-Bench 端到端前置探测（只读）
python3 scripts/ale_scalar_pilot.py                   # ALE 标量评分 pilot（NC 闸不误杀验证）
python3 scripts/check_swe_readiness.py                # SWE-bench Verified 前置探测（只读）
python3 scripts/swe_wiring_pilot.py                   # SWE 接线 pilot（NC 闸二值边界验证）
```

仅用 Python 3 标准库，无依赖、无网络、无 API key。ALE-Bench / SWE-bench 是可选接入层：
本机 + ecs-hz + hpccube 三通道实测端到端缺环（ALE：包缺失/HF 超时/无 docker/无 LLM key；
SWE：swebench 缺失/HF 超时/GITHUB_TOKEN unset）。前置检查如实报 `BLOCKED`，
`fuel_grade=LOCAL_HELD_OUT`（本地 held-out 燃料成立）；两个 pilot 用确定性生成器 +
mock 评分器验证接线与 NC 闸行为（honest PASS / padding OPTIMISTIC_GAP / corrupted
SELECTION_CORRUPTED；SWE 二值 resolved 被 saturation 轴正确拒收——须用 F2P 比例连续分）。

## 结构

```
rsi_agent/
  core.py       ScoreVector / HeuristicPolicy / PublicHistory（无 private 字段）
  tasks.py      ContextualTask：隐藏真值驱动的上下文决策任务
  barriers.py   SealedPrivateSet + assert_public_only 类型屏障
  gates.py      paired_gate / ESS / effect floor / Holm / 回归负担 / PromotionGate
  nc_guard.py   NC 前置 model-free 审计闸（selection_error / signed_gap / saturation）
  audit.py      AuditLog 哈希链 + verify()
  outer.py      InnerAgent：从公开反馈爬升，无隐藏真值访问
  runner.py     RSIRunner 编排 + RunResult 四态判决
  ale_adapter.py  ALE-Bench 接入层（标量评分 → ScoreVector；缺环 fail-closed）
  swe_adapter.py  SWE-bench Verified 接入层（F2P/resolved → ScoreVector；缺环 fail-closed）
run_demo.py     CLI
tests/          41 个反证/负对照
scripts/check_ale_readiness.py   ALE-Bench 接入前置检查（只读）
scripts/ale_scalar_pilot.py      ALE 确定性标量评分 pilot
scripts/check_swe_readiness.py   SWE-bench Verified 前置检查（只读）
scripts/swe_wiring_pilot.py      SWE 接线 + NC 二值边界 pilot
results/demo.json  60 代完整结果
results/ale_pilot.json  ALE pilot 结果
results/swe_pilot.json  SWE pilot 结果
```

## 边界（诚实说明）

这是一个**构造性 + 纪律性**屏障，不是 OS 级安全隔离：进程内恶意代码仍可读 `_SEALED`（测试用检查验证的是"InnerAgent 只收 PublicHistory"这个契约）。真实 RSI 系统需要把 scorer / private set 放进独立进程或 enclave。此 agent 证明的是"**非循环的自我改进 + 可证伪的提升判定**"这一闭环，不是"安全沙箱"。
