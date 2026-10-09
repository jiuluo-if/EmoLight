# 本地数据

此目录不包含录音或数据集。将数据集放在本地独立路径，并先核对其许可与数据处理约束；训练/评估工具不自动下载语料。

训练和评估使用用户准备的 CSV manifest，必需列为 `path,emotion,speaker_id,recording_id`，可选列为 `dataset_id,source_recording_id,augmentation_group_id`。相对路径以 manifest 所在目录为基准。Emotion 支持 neutral/happy/angry/sad；常见 CREMA-D/EmoDB 简写有默认映射，其他标签可通过自定义映射转换，不支持的标签会计数并跳过。

划分器会把同一 dataset 的 speaker、`recording_id`、`source_recording_id`、`augmentation_group_id`，以及跨 dataset 的规范化绝对路径和非空 WAV SHA-256 建成连通组，保证训练、验证、测试不共享来源音频。请为同一原始录音的切片填写相同 source ID，并为增强版填写相同 augmentation group。

训练只使用通过 `ACCEPTABLE` 音频质量门控的窗口；`LOW_QUALITY` 和 `UNCERTAIN` 样本会分别计数并排除。实时与离线评测也拒绝不确定质量窗口，因此没有可用底噪参考的录音需要先按相同采集协议补充安静背景样本，不能把“不确定”当成质量合格。

测试 WAV、生成的合成音频和个人注册录音不要放进版本控制；仓库 `.gitignore` 已忽略音频、声纹参考和模型权重。

需要进行记录噪声评测/训练增强时，可以准备独立 `noise_manifest.csv`，列为 `noise_type,path`；类型可为 `background_music`、`fan`、`environment` 或 `reverb_impulse`。路径按 manifest 所在目录解析，采样率必须与目标模型一致。训练只对 training split 合成白噪声/记录背景噪声或混响；validation/test 保持原样。噪声与房间脉冲响应的许可和录制环境应单独记录，工具不会把合成或未经验证的条件冒充真实房间评估。

当前能量 VAD 与声学统计不构成已验证的人声检测器或情绪检测器。
