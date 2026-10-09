# 模型状态

当前没有声纹或情绪模型权重。`SpeakerVerifier` 默认实现固定返回 `NOT_CONFIGURED`，实时目标情绪链路因此拒识。SVM 适配器、模型校验、训练和评估工具已经提供，但没有真实模型或语料指标。

训练依赖单独安装：`python -m pip install -e ".[ml]"`。本地 WAV manifest 通过 `emolight-train --manifest <csv> --model <local-file>` 训练并保存 `prosody-v1` calibrated LinearSVC artifact。只有明确确认持有许可的真实标签时，才用 `emolight-evaluate --manifest <csv> --model <local-file> --confirm-real-labeled-data` 输出性能指标；不确认时报告 `NOT_EVALUATED_NO_LABELED_DATA`。模型 artifact 使用 joblib；只加载自己或可信训练过程生成的文件，不能加载来源不明的 pickle/joblib。当前仓库没有真实模型权重或可用标注语料。

接入模型前需要在许可允许的数据上训练并保存模型和预处理参数，通过说话人互斥、噪声鲁棒性、拒识和概率校准评估；验证完成后才可以实现真实 `EmotionPredictor` 与 `SpeakerVerifier`。
