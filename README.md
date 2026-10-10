# EmoLight

**Lightweight Target-Speaker Emotion Recognition & Adaptive Ambient Lighting** 的本地 PC 原型。

本仓库当前交付音频 WAV 读取、基础声学特征、能量门限 VAD 接口、未配置声纹/情绪模型时的安全拒识、独立灯光策略、Tkinter RGB 灯带模拟器。它不会把演示数据称为真实识别：演示状态始终标记 `SIMULATION`。

## 当前功能与限制

- 本地 PCM WAV（8/16/24/32 bit）读取、立体声转单声道；计算 RMS、过零率、削波占比和粗略质量标记。
- 25 ms 能量门限 VAD 仅用于数据流展示；它不能区分讲话、音乐和其他响声，也不是目标人检测器。
- `emotion-prosody-24-v1` 提供 24 维非语义韵律特征，并有 8 维 energy/rhythm simple 对照。无可靠 F0/HNR 使用显式缺失值，由训练集 imputation 参数统一处理；有效输入帧、能量活动、周期性和 F0 有效率分开记录。
- 音频质量把电平、削波、活动比例和质量状态分开；只有存在非活动噪声参考帧时才给出粗略前景/底噪能量比。它不是经校准的 SNR；连续噪声/讲话无法估计底噪时状态为 `UNCERTAIN`。
- 可选的 `sounddevice` 麦克风采集会以独立有界队列处理音频帧，GUI 只在 Tk 主线程轮询最新运行结果。采样率、分析帧长/帧移、活动阈值、上下文窗口、刷新间隔和质量门限由 `AppConfig` 统一设置；模型特征契约不匹配时安全拒识。
- 本地 `manifest.csv` 可训练经分组交叉验证校准的 `StandardScaler + LinearSVC` 基线；speaker 与原始 recording 连通分组，不跨训练/验证/测试集合。离线 emotion-only 分类不会验证目标身份，也不会生成目标事件。
- `models/emodb_four_class.json` 是从真实 Berlin EmoDB 1.3.0 训练得到的 24 维 CPU 线性模型；`live_emotion_only.py` 可通过 WAV 或麦克风输出明确标为 `EMOTION_ONLY_EXPERIMENTAL` 的四类分数。该模型 test Macro-F1 为 0.570，happy recall 仅 0.049，不能称为可靠的通用情绪识别。
- 目标说话人 verifier 仍未配置；实验输出不归属到指定人物，不生成 `TARGET_ACTIVE` 事件，不驱动灯光。目标模式继续 fail-closed。
- 模拟模式提供四种演示情绪、模拟分数和灯效。演示事件不能进入真实自动策略。
- 灯光默认暖色/低刺激映射、无闪烁、亮度上限 40%、夜间上限 12%、至少 1.5 秒过渡；手动暖白会关闭自动控制。
- `configs/default.json` 可调整每类情绪的 RGB 颜色、亮度上限、夜间上限和过渡时长；非法颜色、频闪设置和过短过渡会被拒绝。
- 没有连接或验证实体 LED 灯带、串口控制器或 ESP32 固件。

这些灯光规则是可调整的交互策略，不是医疗或心理治疗方案，也不用于推断真实心理状态。

## 安装

需要 Python 3.11 或更新版本。建议在虚拟环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

训练/评测工具使用 scikit-learn，安装 `python -m pip install -e ".[ml]"`。实时与 WAV 推理只使用 NumPy 读取 JSON 模型，不依赖 scikit-learn 或 PyTorch。完整开发测试环境可安装 `python -m pip install -e ".[dev,ml,audio]"`。

需要启用可选麦克风采集时安装 `python -m pip install -e ".[audio]"`。麦克风设备和驱动由本机提供。

Tkinter 通常随 Windows Python 安装；NumPy 用于离线信号特征，pytest 用于测试。运行路径不依赖 PyTorch、ASR 或网络服务。

## 运行

```powershell
# 查看当前真实识别配置；没有模型时输出 NOT_CONFIGURED
python -m emolight --no-gui

# 读取本地 WAV 并输出声学特征；不会预测情绪
python -m emolight --no-gui --wav .\path\to\local.wav

# 输出明确标记的模拟事件和灯光命令
python -m emolight --demo --emotion happy --no-gui

# 使用本地配置文件的音频、模型与灯光参数
python -m emolight --demo --no-gui --config .\configs\default.json

# 对本地 PCM WAV 执行 emotion-only 分类（不会验证 speaker 或产生目标情绪事件）
emolight --no-gui --emotion-only --wav .\clip.wav --model .\models\emodb_four_class.json

# 对许可合规的 speaker-exclusive manifest 训练模型
python .\scripts\train_linear.py --manifest .\manifest.csv --output-dir .\models --augment-white-noise-snr 20 10 5 0 --augmentation-noise-manifest .\noise_manifest.csv --augment-background-snr 20 10 5 0

# pure NumPy emotion-only diagnostics
python .\scripts\predict_linear.py --wav .\clip.wav --model .\models\emodb_four_class.json --config .\configs\default.json

# 实验性实时情绪分数；没有目标身份归属，不触发灯光
python .\scripts\live_emotion_only.py --microphone --model .\models\emodb_four_class.json
python .\scripts\live_emotion_only.py --wav .\clip.wav --model .\models\emodb_four_class.json

# 仅在明确确认真实标注数据及授权后生成 held-out 指标
emolight-evaluate --manifest .\manifest.csv --model .\models\full.json --noise-manifest .\noise_manifest.csv --confirm-real-labeled-data --output .\evaluation.json

# 在完全相同的 held-out 条件下并排比较两个模型；差值为 full - simple
emolight-evaluate --manifest .\manifest.csv --simple-model .\models\simple.json --full-model .\models\full.json --noise-manifest .\noise_manifest.csv --confirm-real-labeled-data --output .\comparison.json

# 打开桌面模拟器；可点选四种模拟情绪、启动麦克风或使用手动暖白
python -m emolight
```

## 真实 EmoDB 基线与局限

数据和模型准备、说话人划分及训练命令见 [data/README.md](data/README.md) 与 [models/README.md](models/README.md)。重建数据集时执行：

```powershell
python scripts/prepare_emodb.py --download --output-dir data/private/emodb-1.3.0/prepared --seed 42 --window-s 1.5 --stride-s 0.5
python scripts/train_linear.py --manifest data/private/emodb-1.3.0/prepared/emodb_manifest.csv --dataset-metadata data/private/emodb-1.3.0/prepared/dataset_metadata.json --output-dir data/private/emodb-1.3.0/trained-models --seed 42 --window-s 1.5 --update-interval-s 0.5 --augment-white-noise-snr 20 10 5 0
Copy-Item data/private/emodb-1.3.0/trained-models/full.json models/emodb_four_class.json
emolight-evaluate --manifest data/private/emodb-1.3.0/prepared/emodb_manifest.csv --model models/emodb_four_class.json --confirm-real-labeled-data --output reports/emodb_1_3_0_evaluation.json
```

当前模型在官方 4-speaker held-out test 上 Macro-F1=0.570、UAR=0.613；angry recall=0.965、happy recall=0.049、neutral recall=0.718、sad recall=0.720。test 统计以相互重叠的 1.5 秒窗口为单位（569 windows / 136 utterances），不是独立 utterance 数。10/5/0 dB white-noise 条件被音质门控全部拒识；20 dB 覆盖率 0.341。背景音乐、风扇、真实环境噪声和混响因没有授权源文件而标记 `NOT_EVALUATED`。完整混淆矩阵、逐类指标和资源数据见 `reports/`。

数据为德语表演情绪语音。该基线不证明自然对话、普通话、其他说话人群体或真实房间环境中的性能；当前 happy 召回率过低，不适合作为可靠情绪判断或目标人物灯光控制依据。

安装后也可使用 `emolight` 命令。桌面 UI 的模拟事件标记为 `SIMULATION`，四类分数和唤醒度仅在模拟事件中显示；麦克风运行时状态栏标记为 `LIVE`，只显示音频质量、活动和身份门控状态，不显示情绪分数。EmoDB 实时情绪分数目前只能通过 `EMOTION_ONLY_EXPERIMENTAL` CLI 路径查看，该路径不验证目标身份、不生成目标事件，也不驱动灯光。

## 测试

```powershell
python -m pytest -q
```

覆盖事件校验、WAV 特征、静音/噪声拒识、数据清单与 speaker 隔离、训练/校准、NumPy 模型加载、实验运行时标签平滑，以及 WAV 实时管线回放。已在本机 Realtek 麦克风执行 5 秒 smoke run；这不构成真实情绪准确率验证。实体灯带仍未连接。

## 模块

```text
src/emolight/
  audio/       WAV、固定长度环形缓冲、可选麦克风适配和简易 VAD
  features/    非语义声学统计
  speaker/     可替换的目标身份接口（当前未配置）
  emotion/     纯 NumPy JSON 情绪推理器
  training/    sklearn 训练、验证集校准、独立测试与噪声实验
  data/        manifest 适配与 speaker/recording 分组切分
  lighting/    事件门控、灯光策略与模拟控制器
  gui/         Tkinter 模拟器
tests/         关键边界与安全行为
configs/       默认音频/灯光参数
  data/          本地清单、许可说明和 split 防泄漏接口（不包含语料）
  models/        当前模型状态与接入条件（不包含权重）
```

## 后续工作

1. 增加有许可的多说话人/多语言数据并保持新的独立测试人群；改进 happy 类泛化后再重新验证。
2. 获取有许可的音乐、风扇、真实环境噪声与房间脉冲响应，完成独立鲁棒性评估。
3. 集成并验证目标说话人模型；未达到门限继续 fail-closed。
4. 评估 MCU/C++/INT8 参数导出和缓冲实现。
5. 若后续接入灯带，再单独验证串口与硬件安全；当前灯光实现冻结。

仓库包含带完整 provenance 的 EmoDB 轻量模型，但其类别性能不足以支持可靠决策；实验 CLI 明确显示 `EMOTION_ONLY_EXPERIMENTAL`。目标说话人 verifier 仍未配置，因此当前不具备经验证的目标人物实时情绪识别能力。
