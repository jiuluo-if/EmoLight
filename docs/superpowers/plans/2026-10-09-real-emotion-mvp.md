# EmoLight Real Emotion MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加可训练、可评估的非语义声学情绪分类，同时确保未通过验证的目标身份不能产生实时目标情绪事件。

**Architecture:** `VAD/质量 -> SpeakerVerifier -> EmotionPrediction -> runtime 构造目标事件`；离线训练/评估复用同一 prosody 特征但只输出 emotion-only 结果。输入、输出缓冲有界且 UI 线程独占 Tk 操作。

**Tech Stack:** Python 3.11+, NumPy, scikit-learn, joblib, pytest, Tkinter；scikit-learn/joblib 使用 `ml` optional extra。

**Spec:** `docs/superpowers/specs/2026-10-09-real-emotion-mvp-design.md`

## Global Constraints
- 不使用 ASR、关键词、文本或语义模型；默认离线；不自动下载数据集。
- 未验证 speaker 时不运行实时情绪预测，也不生成 `TARGET_ACTIVE`。
- 无真实训练语料时，合成数据只用于测试，性能报告标为 `NOT_EVALUATED_NO_LABELED_DATA`。
- 不增加灯光功能；保留旧 CLI、测试与演示模式。
- 固定音频窗口和有界输入/结果队列；模型只加载一次。
- speaker 与原始 recording 不得跨训练/验证/测试集合。

---

### Task 1: Identity gate

**Files:** `src/emolight/events.py`, `src/emolight/speaker/verifier.py`, `src/emolight/emotion/predictor.py`, `src/emolight/runtime/pipeline.py`, `src/emolight/config.py`, `tests/test_runtime_identity_gate.py`

- [x] 写 fake happy predictor 置信度 1.0、verifier 缺失/非目标时不得调用预测器或生成目标事件的测试。
- [x] 分别测试 `SILENCE`、短促瞬态、`LOW_QUALITY`、`OVERLAP` 和 `NOT_CONFIGURED`。
- [x] 扩展 `SpeakerVerification` 与独立 `EmotionPrediction`；runtime 按 VAD→speaker→emotion 执行，并仅由 runtime 合成 `TARGET_ACTIVE`。

### Task 2: Prosody-v1 feature vector and quality gate

**Files:** `src/emolight/features/prosody.py`, `src/emolight/features/acoustic.py`, `src/emolight/audio/vad.py`, `src/emolight/config.py`, `tests/test_prosody.py`, `tests/test_audio_quality.py`

- [x] 合成正弦和频率阶跃验证 F0/变化率；静音验证 F0 缺失与 voiced=0。
- [x] 噪声、削波、NaN/Inf、不同采样率和短帧验证数值稳定性和 schema 元数据。
- [x] 实现固定 32 维窗口向量及版本/采样率/帧长/帧移；扩展旧 `AcousticFeatures` 并保持五参数构造兼容。
- [x] runtime 使用最小活动比例和连续帧条件，不因一个瞬态进入 speaker/emotion 阶段。

### Task 3: Dataset manifest and leakage-resistant splitting

**Files:** `src/emolight/data/manifest.py`, `src/emolight/data/split.py`, `tests/test_manifest.py`, `tests/test_data_split.py`

- [ ] 测试 CSV 标签、路径、speaker/recording ID 校验和缺行处理。
- [ ] 构建 speaker 与 recording ID 的连通分组，再按组划分 train/validation/test。
- [ ] 测试任何 speaker 或 recording ID 均不跨 split；组数不足时明确报错，不回退随机切窗。

### Task 4: Calibrated lightweight classifier and safe model loader

**Files:** `src/emolight/emotion/sklearn_predictor.py`, `src/emolight/training/train_svm.py`, `src/emolight/config.py`, `tests/test_sklearn_predictor.py`

- [ ] 写模型缺失、损坏、类别顺序错误和特征 schema/采样率不一致的拒识测试。
- [ ] 训练 StandardScaler + LinearSVC 与分组交叉验证校准；保存类别、schema、采样参数、版本和 split 摘要。
- [ ] 实现只返回情绪类别/分数的预测器；模型只从可信本地路径加载。
- [ ] 添加可重复训练 CLI；仅按用户指定路径写模型产物。

### Task 5: Metrics, noise and offline emotion-only CLI

**Files:** `src/emolight/training/evaluate.py`, `src/emolight/training/noise.py`, `src/emolight/cli.py`, `tests/test_evaluation.py`, `tests/test_demo_cli.py`

- [ ] 计算 Macro-F1、UAR、逐类 PR、混淆矩阵、校准、拒识率、模型大小、CPU 时延和峰值内存。
- [ ] 为 20/10/5/0 dB 噪声及独立他人语音混合比例生成分开的实验条件。
- [ ] 添加显式 `--emotion-only --model` WAV 命令，不输出目标身份结论。
- [ ] 无真实标注数据时报告 `NOT_EVALUATED_NO_LABELED_DATA`，不报告合成指标。

### Task 6: Latest-only runtime/UI queue and configuration consistency

**Files:** `src/emolight/audio/microphone.py`, `src/emolight/audio/ring_buffer.py`, `src/emolight/runtime/pipeline.py`, `src/emolight/gui/app.py`, `src/emolight/config.py`, `configs/default.json`, `tests/test_microphone_source.py`, `tests/test_runtime_audio.py`

- [ ] 测试麦克风队列过载时淘汰最旧帧、缓冲和 UI 结果队列保持有界。
- [ ] 音频 worker 不接触 Tk API；UI 使用主线程轮询 latest-only 结果队列。
- [ ] 统一读取 sample rate、window、update interval 和帧参数；WAV/model 不匹配时拒识。
- [ ] 测试预测异常后回到安全状态，不阻塞采集帧回调。

### Task 7: Documentation, repository safety and delivery

**Files:** `.gitignore`, `README.md`, `data/README.md`, `models/README.md`, `findings.md`, `progress.md`

- [ ] 忽略录音、声纹注册样本、joblib/pickle/ONNX/深度权重；保留示例配置和说明。
- [ ] 删除开发文档中不必要的本地个人路径和联系信息。
- [ ] 有真实数据则运行训练/评估；否则只运行合成程序测试并明确真实性能未评估。
- [ ] 跑全量 pytest、CLI、打包和代码审查；使用指定邮箱提交并推送任务分支，核对远端 SHA。

## Current Evidence

审计基线 `64897c85` 的旧 pytest 为 22 passed；现有数据/模型目录没有可用于真实性能评估的本地标注资料。
