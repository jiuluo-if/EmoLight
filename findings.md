# Findings & Decisions

## Requirements
- 默认离线；不转写语音、不识别关键词、不调用 LLM 判断情绪。
- 只有身份可靠、音频质量合格且模型已配置时，真实情绪事件才可驱动自动灯光。
- `NOT_CONFIGURED`、`UNCERTAIN`、低质量/重叠输入均 fail-closed。
- 演示事件必须显式显示 `SIMULATION`。
- 灯光默认无闪烁、亮度上限约 40%、过渡至少 1.5 秒、手动优先。

## Research Findings
- 工作区 `F:\codex\emotion` 起初是空目录；现已建立独立 EmoLight Python 项目和 `main` 初始分支。
- 已有相关 EmotiScreen 记忆属于另一目录和另一产品流程，不适合作为此 EmoLight 项目的代码或能力证明。
- Git 邮箱已配置为 `2966684515@qq.com`；GitHub CLI 登录账号为 `jiuluo-if`；没有同名 EmoLight repository。
- 已创建私有 `jiuluo-if/EmoLight` 并把实现提交推送到 `main`；推前后 SHA 已核对一致。

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

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 附件横跨多个后续阶段，无法一次诚实交付完整真实模型与 ESP32 硬件验证 | 首轮交付可运行 PC MVP，并在文档明确阶段缺口 |
| 安装后的 `emolight.exe` 曾报 `cannot import name 'main'` | 增加项目入口 `main()` 并通过入口回归测试和命令验证 |

## Resources
- 用户附件：`C:\Users\联想\.codex\attachments\04062e8e-1b33-43fb-bc0f-c52ab5e6918d\pasted-text-1.txt`
