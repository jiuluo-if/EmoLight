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

## Phase 4 continuation — 2026-10-10
- 当前 Phase 3 worktree 在 pushed branch `feat/phase3-reliability-baselines`，基线 commit `40769dd55b996c41bf3700d43a3fe08564c9cae2`；新建 `feat/live-emodb-inference`，没有切换/改写 main。
- 官方 Zenodo API 记录 7447302 返回 EmoDB 1.3.0、39,981,818-byte zip、CC-BY-4.0 记录许可、MD5 `9d21362dbc5676ef3ab4745d83ced0db`。下载的 MD5 匹配，ZIP CRC 完整；所有 535 个 WAV 经核验为 16 kHz mono PCM16。原始 archive、展开/窗口音频和 manifest 都在 `.gitignore` 排除的 `data/private/`。
- 新增官方表格导入器：只选四类 gold labels，映射 N/F/W/T；A/fear、E/disgust、L/boredom 排除。官方 test speakers 12/14/15/16 原样保留；官方 train speakers 03/09 用于 validation，08/10/11/13 用于 train。固定 1.5 s window / 0.5 s stride，共生成 1400 个本地窗口。cross-split speaker/source/content guard 生效。
- 扩展 manifest 预指定 split 支持；训练 artefact 不保存 test 类别分布，训练预处理、SVC、校准和阈值只使用 train/validation。训练添加 20/10/5/0 dB white-noise augmentation（仅 train），导出 actual 4-class `models/emodb_four_class.json`，size 12,219 bytes / 180 params，NumPy predictor 返回 READY。
- 新增独立 `EMOTION_ONLY_EXPERIMENTAL` runtime 和 `scripts/live_emotion_only.py`。它复用固定环形缓冲/音质拒识/窗口时间戳，但不接 speaker verifier、不创建 EmotionEvent/TARGET_ACTIVE、不会触发灯光。WAV replay 与本机 microphone 入口共享 1.5 s rolling window、0.5 s 更新、20 ms 采集帧和 2-of-3 accepted-label hysteresis。
- full 模型依据 validation 结果预选（correctness-F1 .883 vs simple .721；validation accepted-error .209 vs .426）；其后 official held-out test speakers 未用于阈值/模型选择。最终 test 为 569 overlapping windows / 136 utterances / 4 speakers：clean Macro-F1 .570，UAR .613，coverage .953，accepted-error .315；happy recall .049，明显不足以用于可靠决策。
- 鲁棒性：20 dB white noise coverage .341/Macro-F1 .496；10/5/0 dB 全拒识。其他人 SIR 6/0/-6 dB Macro-F1 为 .402/.192/.111、accepted-error .539/.754/.838。未提供许可音乐、风扇、环境录音或 RIR，所以这些条件如实 NOT_EVALUATED。
- held-out performance run: full JSON 12,219 bytes/180 params; feature median/p95 9.89/14.48 ms; NumPy inference .074/.121 ms; end-to-end 1.61/14.76 ms; CPU 27.47 s, wall 27.67 s, 0.993 cores, peak RSS 153,378,816 bytes。
- 实际 WAV replay 经最终模型成功输出四类分数；5秒本机 Realtek mic smoke run 接通、0 dropped frames，但没有捕获 speech 且全部窗被判 SILENCE，因此不能作为麦克风情绪准确率证据。无音频落盘。
- 最终 fresh 验证：`python -m pytest -q` **107 passed**；`python -m compileall -q src scripts`、`git diff --check`、`emolight-prepare-emodb --help`、`emolight-live-experimental --help`、`emolight-train --help`、`emolight-evaluate --help` 均通过；WAV replay 输出全部四类分数且 mode/identity 明确。`lighting/` 无 diff，data/private WAV/ZIP/CSV 均 ignored。
- 交付：commit `fa31f49e628dcce3d58f3956791b92c8d08b18b1`，作者邮箱按要求设置；分支 `feat/live-emodb-inference` 已推送且远端 SHA 一致。GitHub connector 创建 PR 时返回 403 `Resource not accessible by integration`；已用本机已授权 `gh` CLI 打开 PR 1，状态 OPEN、base main、未合并：https://github.com/jiuluo-if/EmoLight/pull/1。

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
