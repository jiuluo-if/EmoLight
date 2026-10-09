# Findings & Decisions

## Requirements
- 默认离线；不转写语音、不识别关键词、不调用 LLM 判断情绪。
- 只有身份可靠、音频质量合格且模型已配置时，真实情绪事件才可驱动自动灯光。
- `NOT_CONFIGURED`、`UNCERTAIN`、低质量/重叠输入均 fail-closed。
- 演示事件必须显式显示 `SIMULATION`。
- 灯光默认无闪烁、亮度上限约 40%、过渡至少 1.5 秒、手动优先。

## Research Findings
- 当前工作位于 `feat/real-emotion-mvp` 隔离 worktree，基于 main 提交 `5dce973`。
- 新阶段开始时 main 工作区清洁，原测试基线 22 passed；身份门控审计问题确实仍存在。
- `data/` 只有许可说明，没有可用于真实性能评估的标注数据；`models/` 没有模型权重。
- GitHub CLI keyring token 状态无效，但 Git Credential Manager 成功推送任务分支 `124a340`；远端只读读取偶有连接中断，需继续核对 SHA。
- 身份门控修复后，全量测试为 30 passed；假 happy 预测器仅在 verifier 可信返回 TARGET_ACTIVE 后才会被调用，EmotionPrediction 不能编码 speaker/VAD 状态。
- `prosody-v1` 特征和独立音频质量测试已通过；持续高响度噪声但没有非活动噪声参考会留在 UNCERTAIN，不能由 RMS 单独给高质量。
- 新增本地 manifest、speaker/recording 连通组划分、校准线性 SVM 与 emotion-only 评估；`UNCERTAIN`/`LOW_QUALITY` 窗口不进入训练或评测的可用样本，避免训练与部署门控不一致。
- 评审校正：RMS 差分/斜率索引已与发布的特征名对齐；模型元数据现包含活动阈值并在配置、训练和推理中校验；测试集中出现模型未训练的情绪类时，指标明确统计其支持数和零召回，而不是中止评测。
- `joblib` 加载使用 pickle 语义，只应加载本地可信训练产物；仓库无模型文件，文档已保留该信任边界。没有做非可信第三方模型反序列化测试。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Python 3.11+, stdlib + NumPy；pytest 作为开发依赖 | 首版依赖轻、可离线模拟、容易后续替换模块 |
| `AcousticFeatures` 提供 RMS、过零率、帧级统计与明确质量标记 | 能验证本地音频数据流，但不冒充情绪分类 |
| `UnconfiguredEmotionPredictor` 默认输出 `NOT_CONFIGURED`，模拟事件采用不同来源字段 | 防止模拟结果进入真实自动控制链 |
| `EmotionLightingPolicy` 独立于音频与 GUI；仅接收可信结构化事件 | 可单独验证安全策略和后续复用 |
| 麦克风回调只将帧放入有界队列，独立 worker 进入固定长度缓冲区 | 降低采集 callback 被特征计算/GUI 阻塞的风险 |
| 当前使用能量门限 VAD 与未配置 speaker/emotion adapters | 保留可替换契约，模型通过验证前 fail-closed |
| 第一版 GUI 展示模拟状态、灯带、自动/夜间/手动控制 | 确保无麦克风和硬件也能完整运行 |
| 用户颜色、亮度限制和过渡时长从 JSON 读取 | 允许按个人舒适度调整，同时校验频闪、RGB 和过渡安全条件 |
| 新增独立 `EmotionPrediction`；runtime 只在 SpeakerVerifier 可信后包装为 TARGET_ACTIVE | 分类分数不能代替身份验证；旧 `UnconfiguredEmotionPredictor` 兼容层可保留 |
| 使用有 schema/采样元数据的固定 32 维 prosody-v1 向量 | 训练和推理共享特征实现，并可拒绝不匹配模型 |
| 训练产物记录类别顺序、schema、采样参数、划分摘要和版本 | 防止运行时加载含义不一致或数据泄漏模型 |
| 音频质量状态由活动帧、连续活动、削波和有静音噪声参考时的前景/底噪能量差共同判断 | 没有可用底噪参考时标记 UNCERTAIN；`audio_quality` 不再随 RMS 增大而变成高质量 |
| 运行时由单一 `AcousticFeatures` 窗口快照提供活动比例、连续帧及质量门控 | 避免并行 VAD 与质量计算因窗口错位而产生不一致判断 |
| 训练仅接受 `ACCEPTABLE` 音频，评测对不合格/不确定输入计算拒识率 | 训练样本与实际可输出预测的质量域一致，报告不把不确定窗口当作可识别样本 |
| 模型元数据固定活动 RMS 阈值，训练与推理共享完整 `prosody-v1` 参数 | 阈值会改变活动/停顿特征；只比较采样率和帧长不足以保证特征列含义相同 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 附件横跨多个后续阶段，无法一次诚实交付完整真实模型与 ESP32 硬件验证 | 首轮交付可运行 PC MVP，并在文档明确阶段缺口 |
| 安装后的 `emolight.exe` 曾报 `cannot import name 'main'` | 增加项目入口 `main()` 并通过入口回归测试和命令验证 |

## Resources
- 规格来源：本轮用户提供的 EmoLight 下一阶段目标；仓库文档不保留个人本地附件路径。

## Phase 3 Audit Findings
- `origin/main`=`64897c85ecd07a2ab1b1ac1504e59412eec10da6`，匹配用户给定 HEAD。新分支 `feat/phase3-reliability-baselines` 从已推送实现分支切出；共同祖先包含该 main SHA，不会合并 main。
- 检索附件及 `F:\codex` 未找到 `EmoLight_emotion_phase2.patch`。当前没有 `src/emolight/emotion/linear.py`、`scripts/train_linear.py` 或 `scripts/predict_linear.py`；现有为 32 维 prosody + scikit-learn joblib，与要求的 24 维 NumPy 部署接口不等价。
- 新 24 维 extractor 现定义：帧覆盖不足 80%或含 NaN/Inf 即无效；`valid_frame_fraction` 汇总有效分析帧；能量 `active_fraction`、周期性 `mean_periodicity`、F0 `f0_valid_fraction` 明确分开。未检出可靠 F0/HNR 的列保持 NaN，simple/full 两路径均共享训练数据拟合的 imputation。已有程序测试覆盖静音、纯音、低幅男女音域、白噪声和 20/10/5/0 dB 合成噪声的数值行为；非语义特征仍不等同于已确认语音或目标人。
- new `LinearSVC` trainer uses training-only imputation/scaling, one-vs-rest Platt sigmoid on independent validation, Brier/ECE and correctness-F1 rejection threshold; JSON/NumPy inference is integrated in CLI/GUI/runtime, and the joblib path was removed.
- 分组实现已增加 dataset 内 speaker/recording/source-recording/augmentation 连通组、规范化绝对路径、非空 WAV SHA-256；每 split 输出 sample/class/speaker/group/dataset counts 与 missing class 标签。回归测试覆盖同一路径伪造不同 speaker/recording ID、跨路径复制内容和增强组。
- simple/full 同一 split 训练并可独立 held-out evaluation；评估支持 clean、20/10/5/0dB white noise、manifest-based music/fan/environment、RIR reverb、independent other-speaker overlap，并报告 per-stage latency/CPU/RSS/size/params/coverage/error。没有真实语料/噪声源文件时，实际指标仍未评估。
- 仓库没有真实标注音频、声纹注册样本或模型权重；合成测试只能证实程序行为，不能作为情绪性能或环境鲁棒证据。
- simple/full 当前可由同一 evaluator 入口成对评测，固定相同 manifest、seed、噪声清单与条件；结果逐条件给出 `full - simple` 差值。此实现保证测试条件一致，不代表训练出的任一模型已具备有效情绪性能。
- `EmoLight_emotion_phase2.patch` 在可访问附件与仓库位置中未找到；本分支据用户 Phase 3 规格独立实现等价的数据隔离、特征、校准、部署及评测路径。未声称逐行合入或核验该补丁。
- 最终软件回归为 96 passed，CLI/package 入口和静态检查通过。数据集、真实背景噪声源、目标身份验证模型与灯带硬件均不可用，因此真实情绪性能、环境泛化、身份验证与硬件行为没有验证。
