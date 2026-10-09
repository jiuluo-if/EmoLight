# EmoLight PC MVP 设计说明

## 目标与边界

交付一个 Python 3.11+ 的离线桌面 MVP，让开发者能从 WAV 或明确的模拟输入观察声学特征、结构化系统状态与舒适导向的灯光策略。它不声称能识别真实情绪或目标人物：没有实际验证的模型时，预测器必须返回 `NOT_CONFIGURED`；模拟事件始终携带 `SIMULATION` 标记。

本轮实现 PCM WAV 与可选麦克风采集、固定长度滚动缓冲、能量门限 VAD、可解释的 RMS/过零率/质量摘要，情绪事件/灯光策略、模拟灯带及 Tkinter 演示 GUI。声纹注册/验证模型、训练与校准情绪模型、ONNX/TinyTCN、串口硬件和 ESP32 固件不属于本轮交付。

## 架构与数据流

`PCM WAV / bounded microphone frame queue / DemoEventSource -> fixed rolling buffer -> EnergyVAD and acoustic features -> EmotionPredictor -> EmotionEvent -> EmotionLightingPolicy -> LightController -> Tkinter Simulator`

输入、预测、灯光策略、渲染器各自独立。麦克风设备回调只写入有界队列，独立线程再进入固定长度缓冲区和特征模块。`EmotionEvent` 包含来源、状态、情绪置信度、说话人置信度、质量、时间戳和模拟标记。真实自动灯光只有在目标身份与音频质量通过门控、且真实模型已配置并达到置信阈值时才接受事件；当前实现没有这些模型，因此真实事件拒识。模拟场景在 GUI 中明确标识，不能作为真实识别证据。

灯光策略支持 neutral/happy/angry/sad 和拒识回退。默认亮度上限为 40%，夜间更低；过渡时间不少于 1.5 秒；不提供闪烁效果；手动固定颜色和关闭自动模式优先于自动事件。模拟器是硬件控制器的替代接口。

## 技术选择与验证

使用标准库、NumPy、Tkinter；麦克风用可选 `sounddevice`，测试使用 pytest。WAV 读取/特征计算与 GUI 分开；核心策略无需音频设备。测试覆盖缺失模型拒识、非目标/不可靠事件门控、各情绪安全映射、亮度与过渡限值、自动关闭和手动优先。启动说明给出环境安装、headless 演示命令与 GUI 命令。

## 风险和后续阶段

简单声学统计并不能证明情绪分类有效；必须先有合规标注数据、说话人互斥评估、校准与跨噪声结果，才允许接入真实自动情绪事件。Tkinter 设备和窗口可用性取决于运行环境；无显示器时保留 headless 验证。没有灯带硬件时不声明串口/电流安全已验证。
