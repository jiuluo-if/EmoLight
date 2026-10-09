# 模型状态

当前没有声纹或情绪模型权重。`SpeakerVerifier` 默认实现固定返回 `NOT_CONFIGURED`，实时目标情绪链路因此拒识。线性训练与模型校验已提供，但没有真实模型或语料指标。

训练依赖单独安装：`python -m pip install -e ".[ml]"`。`python scripts/train_linear.py --manifest <csv> --output-dir <dir>` 同时训练 8 维 simple 与 24 维 full，训练集拟合缺失值填补/标准化，独立 validation 拟合 Platt sigmoid 并选择拒识阈值，导出 JSON 模型。运行时 predictor 只依赖 NumPy，不反序列化 pickle/joblib。只有明确确认数据真实且有许可时，才使用 `emolight-evaluate --manifest <csv> --model <json> --confirm-real-labeled-data` 输出测试指标；无确认时返回 `NOT_EVALUATED_NO_LABELED_DATA`。当前仓库没有真实模型权重或可用标注语料。

现有训练工具可复现模型拟合、validation calibration 与 held-out evaluation，但真实数据缺失，不能报告 Macro-F1/UAR 或真实噪声鲁棒性。目标 speaker verifier 仍未接入；身份未通过验证时实时链路拒识。
