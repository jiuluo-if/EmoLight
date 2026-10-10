# EmoDB 数据准备

仓库不包含原始录音。`data/private/` 已加入 `.gitignore`，官方压缩包、解压文件、窗口 WAV 与清单都保存在本机，不应提交。

## Berlin EmoDB 1.3.0

指定来源为 [Zenodo record 7447302](https://zenodo.org/records/7447302)。准备工具锁定 `emodb.zip` 的 MD5，并检查 ZIP CRC、官方 gold train/test 表、所有 WAV 的完整性与 16 kHz mono PCM16 格式。也可以手动下载后用 `--archive` 导入：

```powershell
python -m pip install -e ".[ml]"
python scripts/prepare_emodb.py --download --output-dir data/private/emodb-1.3.0/prepared --seed 42 --window-s 1.5 --stride-s 0.5
# 或：python scripts/prepare_emodb.py --archive <本地 emodb.zip> --output-dir data/private/emodb-1.3.0/prepared
```

官方文件名情绪代码映射为 `N=neutral`、`F=happy`、`W=angry`、`T=sad`。`A` 表示 fear，明确排除，不会映射成 angry；本实验只保留四类 gold labels。官方 held-out test speakers 全程隔离；validation 从官方 training speakers 中独立选择两人，其余用于训练。公开结果只包含汇总统计，不列出个人或语料 speaker 标识。每条语句生成相同定义的 1.5 秒窗口，步长 0.5 秒；同语句所有窗口保持相同 `recording_id` 和增强组，绝不跨集合。

Zenodo 记录元数据列出 CC BY 4.0，而 zip 内 audformat metadata 写 CC0-1.0。模型 provenance 为保守起见按 Zenodo record 的 CC BY 4.0 标注，并保留数据集作者署名。该录音集是德语表演情绪语音，不能据此推断普通话、自然对话或真实噪声环境表现。

训练输出 manifest 和 dataset metadata 仅本地生成。背景音乐、风扇、真实环境噪声和房间脉冲响应需要各自授权的 `noise_manifest.csv`（列 `noise_type,path`）；缺少对应文件时报告 `NOT_EVALUATED_NO_SOURCE`，不把合成白噪声称作真实房间实验。
