# EmoLight

**Lightweight Target-Speaker Emotion Recognition & Adaptive Ambient Lighting** 的本地 PC 原型。

本仓库当前交付音频 WAV 读取、基础声学特征、能量门限 VAD 接口、未配置声纹/情绪模型时的安全拒识、独立灯光策略、Tkinter RGB 灯带模拟器。它不会把演示数据称为真实识别：演示状态始终标记 `SIMULATION`。

## 当前功能与限制

- 本地 PCM WAV（8/16/24/32 bit）读取、立体声转单声道；计算 RMS、过零率、削波占比和粗略质量标记。
- 25 ms 能量门限 VAD 仅用于数据流展示；它不能区分讲话、音乐和其他响声，也不是目标人检测器。
- `prosody-v1` 提供 32 维固定特征向量，含 F0、帧级能量、发声/停顿、变化率、过零率和简化 HNR，并记录采样率、帧长、帧移和 schema 版本。
- 音频质量把电平、削波、活动比例和质量状态分开；只有存在非活动噪声参考帧时才给出粗略前景/底噪能量比。它不是经校准的 SNR；连续噪声/讲话无法估计底噪时状态为 `UNCERTAIN`。
- 可选的 `sounddevice` 麦克风采集会以独立有界队列处理音频帧，GUI 只在 Tk 主线程轮询最新运行结果。采样率、分析帧长/帧移、活动阈值、上下文窗口、刷新间隔和质量门限由 `AppConfig` 统一设置；模型特征契约不匹配时安全拒识。
- 本地 `manifest.csv` 可训练经分组交叉验证校准的 `StandardScaler + LinearSVC` 基线；speaker 与原始 recording 连通分组，不跨训练/验证/测试集合。离线 emotion-only 分类不会验证目标身份，也不会生成目标事件。
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

训练/评测工具使用 scikit-learn 和 joblib，安装 `python -m pip install -e ".[ml]"`。完整开发测试环境可安装 `python -m pip install -e ".[dev,ml]"`。

需要启用可选麦克风采集时安装 `python -m pip install -e ".[dev,audio]"`。麦克风设备和驱动由本机提供。

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
emolight --no-gui --emotion-only --wav .\clip.wav --model .\models\local.joblib

# 对许可合规的 speaker-exclusive manifest 训练模型
emolight-train --manifest .\manifest.csv --model .\models\local.joblib

# 仅在明确确认真实标注数据及授权后生成 held-out 指标
emolight-evaluate --manifest .\manifest.csv --model .\models\local.joblib --confirm-real-labeled-data --output .\evaluation.json

# 打开桌面模拟器；可点选四种模拟情绪、启动麦克风或使用手动暖白
python -m emolight
```

安装后也可使用 `emolight` 命令。桌面 UI 的标题与事件栏持续显示 `SIMULATION`，四类分数和唤醒度仅在模拟事件中显示。

## 测试

```powershell
python -m pytest -q
```

覆盖事件校验、WAV 特征、静音质量、环形缓冲、VAD/Speaker 未配置边界、模拟/真实事件隔离、灯光上限、夜间模式和命令行 JSON。UI 构造已做本机检查；真实麦克风设备、模型和灯带硬件尚未验证。

## 模块

```text
src/emolight/
  audio/       WAV、固定长度环形缓冲、可选麦克风适配和简易 VAD
  features/    非语义声学统计
  speaker/     可替换的目标身份接口（当前未配置）
  emotion/     情绪预测接口和校准 SVM 适配器
  training/    本地训练、离线评估和噪声/他人语音实验
  data/        manifest 适配与 speaker/recording 分组切分
  lighting/    事件门控、灯光策略与模拟控制器
  gui/         Tkinter 模拟器
tests/         关键边界与安全行为
configs/       默认音频/灯光参数
  data/          本地数据许可说明（不包含语料）
  models/        当前模型状态与接入条件（不包含权重）
```

## 后续工作

1. 在目标设备上实测麦克风格式、帧丢失、延迟和噪声质量标记。
2. 集成并评估可验证的目标说话人模型；未达到门限继续拒识。
3. 以许可合规、说话人互斥的数据训练/校准轻量情绪基线，再评估 TinyTCN/INT8/ONNX。
4. 实现防重叠的实时事件平滑与声学模型置信门控。
5. 添加限速串口控制器、ESP32-S3 固件和实物硬件安全验证。

仓库当前没有真实标注语料、训练权重或可运行的目标说话人 verifier。可配置本地 SVM 情绪模型，但实时身份门控仍会拒识，直到接入并验证 speaker adapter。未获得真实训练/评估证据前，不应宣称已具备经验证的实时目标人情绪识别能力。
