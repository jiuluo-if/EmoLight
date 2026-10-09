# 本地数据

此目录不包含录音或数据集。将数据集放在本地独立路径，并先核对其许可与数据处理约束；训练/评估工具不自动下载语料。

训练和评估使用用户准备的 CSV manifest，必需列为 `path,emotion,speaker_id,recording_id`，可选列为 `dataset_id`。相对路径以 manifest 所在目录为基准。Emotion 支持 neutral/happy/angry/sad；常见 CREMA-D/EmoDB 简写有默认映射，其他标签可通过自定义映射转换，不支持的标签会计数并跳过。

划分器会把同一 dataset 的 speaker 与 `recording_id` 建成连通组，保证训练、验证、测试不共享 speaker 或原始录音。请为同一录音切分出的所有窗口填写相同 `recording_id`。

训练只使用通过 `ACCEPTABLE` 音频质量门控的窗口；`LOW_QUALITY` 和 `UNCERTAIN` 样本会分别计数并排除。实时与离线评测也拒绝不确定质量窗口，因此没有可用底噪参考的录音需要先按相同采集协议补充安静背景样本，不能把“不确定”当成质量合格。

测试 WAV、生成的合成音频和个人注册录音不要放进版本控制；仓库 `.gitignore` 已忽略音频、声纹参考和模型权重。

当前能量 VAD 与声学统计不构成已验证的人声检测器或情绪检测器。
