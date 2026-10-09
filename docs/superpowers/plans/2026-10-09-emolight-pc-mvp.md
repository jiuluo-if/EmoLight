# EmoLight PC MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个可离线运行并诚实标注真实模型未配置的情绪灯光 PC MVP。

**Architecture:** WAV/有界麦克风帧、固定滚动缓冲、结构化情绪/身份事件、灯光策略与 Tkinter 模拟器分层。未配置预测器拒识；只有可信的真实事件可走自动灯光，演示事件始终是 `SIMULATION`。

**Tech Stack:** Python 3.11+, NumPy, Tkinter, optional sounddevice, pytest。

**Spec:** `docs/superpowers/specs/2026-10-09-emolight-pc-mvp-design.md`

## Global Constraints
- 禁止 ASR、转写、关键词、语义或 LLM 情绪判断。
- 不配置已验证模型时状态为 `NOT_CONFIGURED`。
- 非目标人、低质量、重叠或不确定输入 fail-closed。
- 默认无频闪；最大亮度 40%；过渡不少于 1500 ms；手动控制优先。
- 模拟结果需显式显示 `SIMULATION`。

---

### Task 1: Python package and domain contracts

**Files:** `pyproject.toml`, `src/emolight/__init__.py`, `src/emolight/events.py`, `src/emolight/config.py`, `tests/test_events.py`

- [x] 先写事件、状态和默认配置约束测试。
- [x] 运行 pytest 确认新增测试因模块缺失失败。
- [x] 实现 `SystemStatus`、`Emotion`、`EmotionEvent` 和 `LightingConfig`。
- [x] 运行测试确认通过，并验证 Python 3.11+ 安装元数据。

### Task 2: Offline audio features and fail-closed prediction

**Files:** `src/emolight/audio/wav.py`, `src/emolight/audio/vad.py`, `src/emolight/audio/ring_buffer.py`, `src/emolight/audio/microphone.py`, `src/emolight/features/acoustic.py`, `src/emolight/runtime/pipeline.py`, `src/emolight/speaker/verifier.py`, `src/emolight/emotion/predictor.py`, `tests/test_audio_and_predictor.py`, `tests/test_runtime_audio.py`, `tests/test_microphone_source.py`, `tests/test_vad_speaker.py`

- [x] 测试 WAV PCM 读取、RMS/过零率、低质量标记和未配置预测。
- [x] 验证测试先因目标接口缺失而失败。
- [x] 实现本地 WAV reader 与 NumPy 特征摘要，不执行转写/模型推测。
- [x] 再运行相应测试确认通过。

### Task 3: Safe lighting policy and simulator controller

**Files:** `src/emolight/lighting/policy.py`, `src/emolight/lighting/controller.py`, `tests/test_lighting_policy.py`

- [x] 测试身份/质量/来源门控、四类情绪映射、限亮、夜间、过渡、自动关闭与手动优先。
- [x] 确认测试先失败，再实现策略与 Mock controller。
- [x] 验证所有自动命令遵守配置的亮度和过渡下限。

### Task 4: Explicit demo, CLI, Tkinter simulator and docs

**Files:** `src/emolight/demo.py`, `src/emolight/gui/app.py`, `src/emolight/cli.py`, `README.md`, `configs/default.json`, `data/README.md`, `models/README.md`

- [x] 测试 demo event 带 `SIMULATION`、CLI 状态检查与 headless JSON 输出。
- [x] 实现演示入口、Tkinter GUI、可选麦克风帧队列与固定长度滚动特征；默认命令不要求音频设备。
- [x] 启动 CLI 演示并确认 NOT_CONFIGURED 与 SIMULATION 可区分。
- [x] 完成 README 安装、命令、限制与后续真实模型步骤。

### Task 5: Final review and delivery

**Files:** 所有本计划变更

- [x] 跑用户明确要求的 pytest 与 CLI 检查。
- [x] 检查 diff、秘密信息、协议边界和中文文档。
- [ ] 使用仓库提交策略指定的邮箱，并采用 `English: 中文内容` 提交说明。
- [ ] 配置/确认 GitHub 远端，推送并核对远端 SHA。

## Coverage Notes

麦克风适配器和 GUI 实时音频接线已实现，但实际采集设备未验证。实际声纹模型、真实情绪模型、情绪训练/校准、串口、ESP32 固件和灯带硬件验证仍是后续阶段，不在本计划中声称完成。
