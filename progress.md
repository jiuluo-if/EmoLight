# Progress Log

## Phase 3 continuation — 2026-10-09
- 读取用户新提供的 Phase 3 规格；冻结灯光实现范围。
- 核实 `origin/main`=`64897c85ecd07a2ab1b1ac1504e59412eec10da6`，与规格一致；本地 main 有一个仅 `.gitignore` 的提交，未改写该分支。
- 在干净的既有 feature worktree 新建 `feat/phase3-reliability-baselines`，祖先包含指定 main SHA。
- 在 attachments 与 `F:\codex` 索引中未发现 `EmoLight_emotion_phase2.patch`；已向用户请求该补丁，同时继续独立审计。
- 当前实现与指定补丁不等价：现有 32 维 prosody/sklearn joblib，缺 24 维 pure NumPy `linear.py`/scripts、validation-only threshold/calibration 及 simple/full 同划分对照。
- 修改前全量基线：`python -m pytest -q` → **71 passed**；该基线建立在 phase3 feature 分支尚未修改源代码时。
- Phase 3 TDD data step: 新增 manifest hash/lineage、同一 WAV 不同 speaker/recording ID、复制文件/增强谱系不跨组、每 split 类别与 speaker 分布测试；先运行失败，再实现 union-find 和摘要。聚焦数据测试：**9 passed**。
- 新增实现：`emotion-prosody-24-v1`/`energy-rhythm-8-v1` extractor、有效帧和 F0 missingness、NumPy-only JSON predictor、training-only standardization/imputation、validation sigmoid calibration 与 correctness-F1 threshold 选择、同 split simple/full trainer。后续已接入 CLI/GUI/runtime 并移除 joblib 路径。
- 评估器现按用户确认 guard 独立生成 clean/noise/overlap/reverb metrics；训练不读取 test audio/features，只保存 split 计数。evaluation result includes per-stage latency, CPU, peak RSS, model size/parameter count, coverage, selective error and per-class metrics.
- Mic callback frames carry monotonic arrival timestamp; stop clears pending queue and joins the worker. GUI ignores frames from closed/previous capture sessions, polls results only on Tk main thread and skips older timestamps.
- 当前全量 pytest 最近一次：**92 passed**；之后仍有小范围配置/metadata/test 改动，需最终重新跑全量测试和CLI/package verification。当前没有真实数据/噪声/speaker encoder，故没有真实情绪性能或房间鲁棒结论。
- 完成 simple/full 同 manifest、同 seed、同噪声与 held-out 条件评估入口；报告逐条件 `full - simple` 的 Macro-F1/UAR/coverage/rejection/selective-error 差值。追加 1 项回归后全量测试为 **96 passed**。
- 包入口验证首次发现当前环境遗留旧 `emolight-train -> train_svm` entry point；`pip install --no-deps -e .` 按当前 pyproject 刷新后，entry point 指向 `train_linear:main`，训练和评估 CLI 的 `--help` 均可启动。
- 最终验证：`python -m pytest -q` **96 passed**；`python -m compileall -q src scripts`、`git diff --check`、`emolight --no-gui`、`emolight-train --help`、`emolight-evaluate --help` 均通过。未用合成数据声称真实性能，未验证真实麦克风/真实噪声/灯带硬件。
- 灯光模块 `src/emolight/lighting/` 未修改。Phase 3 分支待提交并推送；不合并 main。

## Session: 2026-10-09

### Current Status
- **Phase:** 4 - Testing & Verification
- **Started:** 2026-10-09

### Actions Taken
- 阅读 EmoLight 完整需求附件。
- 确认目标目录为空；核对全局 Git 邮箱和 GitHub CLI 存在。
- 初始化本任务 `task_plan.md`、`findings.md`、`progress.md`。

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| 目录/Git 状态检查 | 识别既有工程与远端 | 空目录、未初始化 Git | PASS |

### Errors
| Error | Resolution |
|-------|------------|
| using-superpowers 初始路径错误 | 已查找并读取正确 skill 路径 |

## Implementation Log

- 初始化 EmoLight 包、默认配置、离线 WAV/声学摘要、能量门限 VAD、speaker/emotion fail-closed interfaces、固定环形缓冲和可选 `sounddevice` 麦克风源。
- 实现自动灯光门控、模拟/手动控制、渐变 GUI、用户颜色配置和 headless JSON CLI。
- 按 TDD 增加关键模块测试；发现已安装 `emolight.exe` 入口缺少 `main()` 后，新增失败测试并修复。
- GUI 窗口构造、CLI 模拟与未配置输出、偏好配置加载已验证。实际麦克风未打开，未连接硬件，没有训练模型。
- 安装后 CLI 曾因缺少 `main()` 启动失败；已修复并验证 `emolight --no-gui`。
- 最终验证：`python -m compileall -q src tests` 通过；`python -m pytest -q` 为 22 passed；headless LIVE/SIMULATION/config 命令通过；Tkinter 自动/夜间/手动交互冒烟通过；`git diff --check` 通过（仅 Git 的 LF/CRLF 提示）。
- 上一阶段实现提交 `187e45bd29772bad307e168efce92de67fdae3a8` 已推送到项目远端 main。

## Real Emotion MVP Stage

- 当前 worktree：`.worktrees/real-emotion-mvp`，branch `feat/real-emotion-mvp`，基于本地 main `5dce973`。
- 已复核：runtime 没有身份 verifier；任意单帧活动即可调用 predictor；完整 EmotionEvent 可由 predictor 返回。新阶段初始测试基线 22 passed。
- 数据检查：没有音频 manifest 或模型权重。真实性能训练须待用户提供许可合规标注语料。
- GitHub：任务分支 `124a340` 已成功推送；Git CLI keyring 检查仍报 token 无效，但 Git Credential Manager 完成 push。远端只读 `ls-remote` 最近一次连接中断，后续再核 SHA。
- Phase 6.1 completed: EmotionPrediction no longer contains identity status; runtime now owns TARGET_ACTIVE event construction after activity, quality and speaker gates. Compatibility tests pass; full suite is 30 passed.
- Phase 6.2 implemented `prosody-v1` fixed 32-D causal window features with F0/autocorrelation, energy, voiced/pause/activity, deltas, ZCR and HNR metadata. AcousticFeatures retains its original five positional fields and now separates RMS, clipping, activity ratio, VAD continuity and an SNR estimate; no inactive noise floor means quality is UNCERTAIN.
- Synthetic feature and quality tests cover tone, pitch step, silence, noise, nonfinite samples, short frames, clipping and SNR contrast.
- `RealtimeFeatureRuntime` now uses the activity ratio/continuity and quality status from its single shared audio snapshot instead of running a second independent VAD pass.
- Phase 6.2 full suite after feature/quality changes: 38 passed.
- Added local CSV manifest mapping, speaker/recording connected-group splitting, calibrated StandardScaler + LinearSVC training, model metadata validation, offline emotion-only WAV classification, calibration/classification metrics, and noise/interferer evaluation conditions. No dataset/model weights exist in the repository, so real performance remains unevaluated.
- Wired `AppConfig` into microphone, runtime, WAV quality analysis, and GUI model loading. Audio worker publishes a bounded latest-only `RuntimeSnapshot`; only the Tk main thread polls and renders. GUI explicitly reports an unconfigured speaker adapter and invalid emotion model state.
- Code review found and fixed a mislabeled energy delta/slope mapping, missing activity-threshold compatibility metadata, training on windows the deployment-quality gate would reject, and evaluation failure when test labels were absent from model classes.
- Targeted verification after these changes: 71 passed across the affected runtime, feature, training/evaluation, CLI, configuration and safety modules; `compileall`, `git diff --check`, and installed `emolight`, `emolight-train --help`, `emolight-evaluate --help` checks passed. Full suite was not run per repository instruction.
- Actual microphone, real labeled data, trained real-world weights, speaker adapter, and hardware were not available; no real performance claim is made.
- Delivery commit `ef93059365d9daac549cd065be17e7cb11a3b7a5` was pushed to `origin/feat/real-emotion-mvp`; `git ls-remote` SHA matched local HEAD and the worktree was clean.
