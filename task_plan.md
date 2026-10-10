# Task Plan: EmoLight PC MVP and Real Emotion MVP

## Goal
在现有 EmoLight PC MVP 上增量建立可真实评测的非语义声学情绪分类与严格目标身份门控，同时保持默认拒识和原 CLI/演示兼容。

## Next Step
找到/收到 Phase 2 补丁并检查兼容边界；若无法取得，按 Phase 3 规格实现 24 维纯 NumPy 模型契约。

## Current Phase
Phase 7: Phase 3 reliability audit

## Phases

### Phase 1: Requirements & Discovery
- [x] 阅读用户附件并确认工作区为空
- [x] 划定首个交付：本地声学特征接口、NOT_CONFIGURED 识别状态、安全灯光策略、Tkinter 演示器
- [x] 记录外部依赖和未实现能力
- **Status:** complete

### Phase 2: Planning & Structure
- [x] 写设计说明与逐步实施计划
- [x] 建立可安装的 Python 3.11+ 包和默认配置
- **Status:** complete

### Phase 3: Core Implementation
- [x] 先写并运行结构化事件、灯光门控和安全边界测试
- [x] 实现声学特征、未配置模型拒识与灯光控制器
- [x] 建立明确标记 SIMULATION 的桌面演示入口
- [x] 加入可选麦克风队列、固定缓冲、运行特征管线和用户可编辑灯光配置
- **Status:** complete

### Phase 4: Testing & Verification
- [x] 运行用户要求的自动化测试与启动检查
- [x] 检查文档、提交配置与 diff
- **Status:** complete

### Phase 5: Delivery
- [x] 使用约定邮箱与“English: 中文内容”提交
- [x] 推送 GitHub 并核对远端提交
- [x] 总结实现范围、证据和后续阶段
- **Status:** complete

## Delivery
- Previous implementation commit: `187e45bd29772bad307e168efce92de67fdae3a8`

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 空目录中建立独立 `emolight` Python 包 | 当前没有既存代码可迁移 |
| 缺失训练权重/目标声纹时返回 `NOT_CONFIGURED` / `UNCERTAIN` | 禁止编造真实情绪或身份判断 |
| 首版保留 WAV、可选麦克风和 Tkinter 模拟；模型训练、可验证声纹、串口固件列为后续 | 先交付可运行且不会误报能力的 PC MVP |
| 音频处理仅计算可解释声学统计，不做 ASR 或语义判断 | 遵守离线、非语义识别边界 |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| 初次使用了错误的全局 skill 路径 | 改为正确的已安装 skill 路径 |
| 当前目录不存在 Git 元数据 | 确认工作区为空；项目初始化后检查远端设置 |
| 更新设计文档时补丁因原句略有差异未匹配 | 读取文档现状后按实际文本重试 |
| 两次补丁未匹配当前文档/代码行 | 重新读取文件，按现状更新；没有部分写入 |
| console script 曾因缺少 `main()` 启动失败 | 新增入口测试和 wrapper，安装后的命令已通过 |
| 本阶段多文件补丁因表格上下文不一致未应用 | 检查文件后拆分编辑，没有部分写入 |
| GitHub HTTPS 推送被连接中断且 CLI keyring token 无效 | 本地提交保留；继续核对认证通道和远端状态 |
| Phase 3 计划补丁尝试针对 main 旧 task_plan 文本匹配失败 | 重新读取当前 worktree 的 Phase 6 计划，改为增量追加 Phase 7；没有写入部分内容 |

## Phase 6: Real Emotion MVP

### Phase 6.1: Baseline and identity gate
- [x] 复核当前 main 基线、清洁状态和审计缺口
- [x] 现有测试基线：22 passed
- [x] 运行时按 VAD→身份验证→情绪预测，未验证身份时不调用预测器
- [x] 为 NON_TARGET、SILENCE、OVERLAP、LOW_QUALITY、NOT_CONFIGURED 增加状态测试
- [x] 阻止 EmotionPrediction 携带身份/VAD 状态
- [x] 保留旧 WAV CLI JSON 与未配置预测器状态访问兼容
- [x] 全量测试：30 passed
- **Status:** complete

### Phase 6.2: Prosody feature contract
- [x] 新增有 schema/version/frame metadata 的固定 32 维因果韵律向量
- [x] 添加正弦、静音、噪声、异常浮点、削波和帧边界测试
- [x] AcousticFeatures 旧 API 保持兼容，质量/活动/削波/SNR 估计分开
- [x] runtime 以有效活动比例和最长连续帧数拒绝孤立瞬态
- [x] 音频质量状态区分有效底噪估计、低质量和未知；无噪声参考不以 RMS 代替质量
- [x] 全量测试 38 passed
- **Status:** complete

### Phase 6.3: Trainable CPU baseline
- [x] 添加 manifest dataset adapter 和 speaker/recording-connected split
- [x] 添加 StandardScaler + calibrated LinearSVC 训练、元数据和安全加载
- [x] 模型缺失/损坏/schema/class mismatch 均 fail-closed
- [x] 添加显式离线 emotion-only CLI，不生成目标身份结论
- **Status:** complete

### Phase 6.4: Evaluation and robustness
- [x] 训练、验证和测试工具输出 Macro-F1、UAR、每类 PR、混淆矩阵、校准指标
- [x] 支持 20/10/5/0 dB 噪声和独立他人语音混入比例实验
- [x] 记录划分谱系、模型大小、CPU 延迟和峰值内存；合成数据只测流程
- [x] 没有真实标注数据时，标记真实训练/性能未完成，不伪造指标
- **Status:** complete

### Phase 6.5: Quality, runtime, and delivery
- [x] 最小活动比例及连续帧门控；同一窗口快照共享质量和预测
- [x] 统一音频采样率/窗口/刷新配置；结果队列 latest-only 且有界
- [x] 补充 .gitignore、README、移除文档中的本地个人路径
- [x] 选择性回归测试、代码审查、提交并推送任务分支
- **Status:** complete

### Phase 6 verification notes
- [x] 基础回归与新增功能相关测试：71 passed（选择性测试，未运行全量测试套件）
- [x] `compileall` 与 `git diff --check`
- [x] `emolight --no-gui`、`emolight-train --help`、`emolight-evaluate --help`
- [x] 评审发现的特征索引、quality 域、时序阈值和未知类别指标问题已修复并有测试
- [x] 配置 NaN/Infinity 校验与回归测试
- [x] 代码复查与发现问题修复完成
- [x] 提交、推送与远端 SHA 核对；当前实现提交 `ef93059365d9daac549cd065be17e7cb11a3b7a5`

## Phase 7: Phase 3 reliability audit

### Audit baseline
- [x] 读取 Phase 3 规格、README、Git 状态和远端 main
- [x] 确认 `origin/main`=`64897c85ecd07a2ab1b1ac1504e59412eec10da6`
- [x] 确认工作区干净并切到新分支 `feat/phase3-reliability-baselines`
- [x] 核验当前代码是 32 维 + sklearn joblib，而非点名的 24 维纯 NumPy linear/script 补丁
- [x] 检索未找到 `EmoLight_emotion_phase2.patch`；按明确规格独立实现并记录此限制
- **Status:** in_progress

### Implementation
- [x] 按绝对路径、内容哈希、dataset-scoped speaker/recording 和 augmentation lineage 防跨 split
- [x] 输出 train/validation/test 的样本数、类别分布和 speaker 数；不完整类别显式警告
- [x] 定义有效帧/F0 可信度/periodicity/voice 的独立语义并集成 24-D extractor 到 prosody/CLI/runtime
- [x] 在 training split 拟合 scaler/imputer；validation split 校准分数和选择拒识阈值；test split 仅最终评测
- [x] 同一分组切分训练 simple 与 full 非语义特征线性模型
- [x] 增加背景音乐、风扇、混响和重叠人声独立评估与 coverage/reject/error 指标；无源明确 NOT_EVALUATED
- [x] 导出 JSON NumPy 纯推理器，报告模型参数/文件大小、特征/VAD/总延迟和 CPU/RSS
- [x] 集成 `emotion-prosody-24-v1` 与 NumPy predictor 到 CLI/GUI/runtime 并移除 joblib 主链路
- [x] 实时 timestamp/迟到结果/有界队列/线程释放/config gates；保持 lighting 实现冻结
- [x] 全量回归、CLI/package 启动检查与审计报告；独立分支推送，不合并 main
- **Status:** complete

## Phase 8: Real EmoDB model and live experimental inference

### Baseline and data source
- [x] Read Phase 4 user specification and verify current feature branch/worktree before edits
- [x] Create isolated branch `feat/live-emodb-inference` from the pushed Phase 3 branch; do not change main or lighting
- [x] Verify official audEERING EmoDB metadata: CC0-1.0, German acted speech, 16 kHz mono PCM, 10 speakers, and published gold train/test tables
- [x] Query the named Zenodo record/API for exact file, checksum, license and availability; acquire data only into ignored local data storage
- [x] Implement dataset manifest import with correct N/F/W/T labels, speaker ids, path normalization, integrity/audio checks and duplicate rejection
- [x] Design a speaker-exclusive training/validation split over official train speakers while preserving the official held-out speaker test split

### Model and runtime
- [x] Test-first implementation for bounded 1.5 s feature windows and training/inference parity
- [x] Train the existing lightweight feature + LinearSVC pipeline on real four-class EmoDB utterances; reject ambiguous/unmapped classes
- [x] Calibrate probabilities/rejection threshold using only validation speakers; never tune on official test speakers
- [x] Export and load actual trained JSON weights; verify metadata and class order; keep data and weights ignored locally
- [x] Add live `EMOTION_ONLY_EXPERIMENTAL` microphone mode with 16 kHz mono, 1.5 s rolling window, 0.5 s updates, bounded queues, smoothing and timestamp/status output
- [x] Add WAV replay through the same rolling-window predictor path; retain TARGET_CONDITIONED_LIVE fail-closed without verifier
- [x] Measure model size/parameter count, CPU feature/inference and end-to-end latency, queue drops, peak RSS; report honest held-out metrics and acted German domain limits

### Final verification and delivery
- [x] Run relevant tests first, then the complete existing suite and WAV replay checks
- [x] Confirm no lighting source changes and no dataset/raw audio committed
- [x] Update README/data/model usage, results report and limitations
- [x] Commit with required author and `English: 中文` message, push branch, verify remote SHA, open PR without merging main
- **Status:** complete
