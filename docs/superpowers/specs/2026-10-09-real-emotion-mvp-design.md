# EmoLight 真实情绪 MVP 设计

## 目标和边界

为现有离线 PC 原型增量建立真实可训练、可评测的非语义声学分类链路。只有实时活动比例和连续帧条件满足、目标身份验证返回可信 `TARGET_ACTIVE`、且音频质量合格时，运行器才调用情绪分类器并生成目标情绪事件。情绪分类器只产生独立 `EmotionPrediction`，不能设置 speaker 状态或构造目标事件。训练/离线评估允许只分类情绪，但输出必须标记为 emotion-only。

没有真实声纹适配器时，speaker verifier 返回 `NOT_CONFIGURED`，运行器不调用情绪分类器。当前没有标注语料或权重，因此首轮完成适配器、训练和评估命令不等于真实模型已训练；禁止报告合成数据性能为情绪识别性能。

## 架构和数据流

实时路径：

`AudioFrame -> fixed ring buffer -> energy activity/quality gate -> SpeakerVerifier -> EmotionPredictor -> RuntimeSnapshot -> bounded latest-only UI queue`

离线实验路径：

`manifest -> connected speaker/recording group split -> shared prosody-v1 extractor -> StandardScaler + calibrated LinearSVC -> held-out evaluation`

目标事件由 runtime 包装。speaker 与 emotion 预测结果是独立数据类型。适配器缺失、载入错误或 schema 不匹配均拒识。音频帧回调只入有界队列，特征/分类在线程 worker 执行；UI 线程轮询有界结果队列并操作 Tk 控件。灯光策略和现有灯光 GUI 不扩展。

## 特征和分类

新 `prosody.py` 提供 32 个基于 F0、帧级能量、活动/停顿、变化率、过零率及简化自相关声质的固定窗口统计。每组向量记录 schema 版本、采样率、帧长和帧移；训练和推理调用同一提取器。只使用已到达的帧；静音、削波和异常浮点输入经过显式门控。

音频质量将 RMS 电平、削波占比、活动比例和状态独立记录。只有存在非活动帧时才估算 active/inactive 能量对比；该值只是粗略估计，不是校准过的 SNR。没有可用噪声参考时必须保持 `UNCERTAIN`，不能用响度代替质量。

情绪类别为 neutral/happy/angry/sad，实际训练允许语料只有其中子集。SVM 训练保存 StandardScaler、校准分类器、类别顺序和元数据。真实模型文件只从用户本地数据路径生成，不自动下载。joblib 模型文件视为可信本地产物，不应加载第三方不可信文件。

## 数据划分和评估

Manifest 至少含 path、emotion、speaker_id、recording_id。按 speaker 与原始录音的连通分组划分，任一 speaker 或 recording 不得跨 train/validation/test。报告 Macro-F1、UAR、逐类 Precision/Recall、混淆矩阵、校准误差、拒识率、模型字节数、CPU 推理时延及峰值内存。噪声实验以 20、10、5、0 dB 添加白噪声；其他说话人语音混入比例另列，不能用普通降噪代表目标语音分离。

无真实标注样本时，只验证适配器和脚本；合成信号仅用于特征与程序正确性测试，真实指标状态为 `NOT_EVALUATED_NO_LABELED_DATA`。普通话声调会影响 F0 变化解释，需在报告中作为限制。

## 配置和兼容

扩展现有配置对象时保留 `LightingConfig` 与当前 CLI 参数。实时采样率、分析窗口、刷新间隔、VAD 活动比例均来自同一 `AppConfig`。WAV 的采样率与模型元数据不匹配时拒识，不隐式错误重采样。保留现有模拟灯效和 JSON 输出字段。
