# 情绪模型

`emodb_four_class.json` 是用真实 Berlin EmoDB 1.3.0 训练、由独立 validation speakers 校准的 24 维 LinearSVC + NumPy 推理模型。它包含四类参数、训练集标准化/缺失值填补、validation 校准器/拒识阈值、窗口协议和数据来源。模型为 11,949 bytes、180 个参数；运行时只需 NumPy。公开元数据只保留数据来源和汇总划分信息，不列出语料 speaker 标识。训练与 held-out 结果见 [`reports/emodb_1_3_0_baseline.md`](../reports/emodb_1_3_0_baseline.md) 和 JSON 报告。

## 实验边界

该模型是 `EMOTION_ONLY_EXPERIMENTAL`。EmoDB 是德语表演语音；当前 held-out 测试显示类别泛化不足，尤其 happy recall 很低。它不能代表自然对话、普通话或真实房间抗噪能力，也不识别指定说话人。

目标说话人验证器仍是 `NOT_CONFIGURED`。现有 target-conditioned runtime 会继续拒识；实验 CLI 不创建 `TARGET_ACTIVE` 事件，不触发灯光。正式灯光策略保持冻结。

## 训练和运行

准备数据后，以下命令可重建模型：

```powershell
python scripts/train_linear.py --manifest data/private/emodb-1.3.0/prepared/emodb_manifest.csv --dataset-metadata data/private/emodb-1.3.0/prepared/dataset_metadata.json --output-dir data/private/emodb-1.3.0/trained-models --seed 42 --sample-rate 16000 --frame-ms 25 --hop-ms 10 --window-s 1.5 --update-interval-s 0.5 --augment-white-noise-snr 20 10 5 0
Copy-Item data/private/emodb-1.3.0/trained-models/full.json models/emodb_four_class.json
```

WAV 流程（CI/无麦克风）和实时麦克风（需 `.[audio]`）共用相同的 1.5 秒环形窗口、0.5 秒更新与 NumPy predictor：

```powershell
python scripts/live_emotion_only.py --wav .\clip.wav --model .\models\emodb_four_class.json
python scripts/live_emotion_only.py --microphone --model .\models\emodb_four_class.json
```

输出带 `EMOTION_ONLY_EXPERIMENTAL`、四类分数、音频活动/拒识原因和时间戳；标签仅对连续合格预测做 2-of-3 平滑。没有身份验证，不应把输出归因于指定人物。

JSON 中 `emotion` 是平滑标签，`candidate_emotion`、`scores` 和 `confidence` 表示当前窗口的原始候选；`confidence_for` 标明该置信度所属类别。拒识时不会延续上一标签。

数据集来源和署名：Burkhardt et al., Berlin EmoDB 1.3.0, [Zenodo 7447302](https://doi.org/10.5281/zenodo.7447302), CC BY 4.0（Zenodo 记录许可；archive 内部 metadata 另写 CC0-1.0）。原始数据不随仓库分发。
