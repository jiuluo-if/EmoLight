import time
import tkinter as tk
from tkinter import ttk
from dataclasses import replace

from emolight.audio.microphone import MicrophoneAudioSource
from emolight.config import AppConfig, LightingConfig
from emolight.demo import make_demo_event
from emolight.emotion.predictor import ModelStatus, UnconfiguredEmotionPredictor
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.events import Emotion
from emolight.lighting.controller import SimulatorLightController
from emolight.lighting.policy import EmotionLightingPolicy, LightCommand
from emolight.runtime.pipeline import RealtimeFeatureRuntime, RuntimeSnapshot
from emolight.runtime.result_queue import LatestOnlyQueue
from emolight.speaker.verifier import UnconfiguredSpeakerVerifier


class EmotionSimulatorApp:
    def __init__(self, root: tk.Tk, config: AppConfig | LightingConfig | None = None) -> None:
        self.root = root
        self.root.title("EmoLight | 本地灯光模拟器")
        self.root.minsize(620, 470)
        app_config = config if isinstance(config, AppConfig) else AppConfig(lighting=config or LightingConfig())
        self.config = app_config
        self.policy = EmotionLightingPolicy(app_config.lighting)
        self.controller = SimulatorLightController()
        self.automatic = tk.BooleanVar(value=True)
        self.night = tk.BooleanVar(value=False)
        self.manual_override = False
        self.mic_source: MicrophoneAudioSource | None = None
        self._mic_session_id = 0
        predictor = UnconfiguredEmotionPredictor()
        self.model_load_error: str | None = None
        if app_config.emotion_model_path:
            try:
                predictor = NumpyLinearEmotionPredictor.load(
                    app_config.emotion_model_path,
                    expected_sample_rate=app_config.audio.sample_rate_hz,
                    expected_frame_ms=app_config.audio.frame_ms,
                    expected_hop_ms=app_config.audio.hop_ms,
                    expected_activity_rms_threshold=app_config.audio.activity_rms_threshold,
                )
            except (OSError, ValueError, RuntimeError) as error:
                self.model_load_error = str(error)
        audio = app_config.audio
        self.runtime = RealtimeFeatureRuntime(
            sample_rate=audio.sample_rate_hz,
            window_s=audio.window_s,
            update_interval_s=audio.update_interval_s,
            predictor=predictor,
            speaker_verifier=UnconfiguredSpeakerVerifier(),
            frame_ms=audio.frame_ms,
            hop_ms=audio.hop_ms,
            activity_rms_threshold=audio.activity_rms_threshold,
            clipping_limit=audio.clipping_limit,
            minimum_snr_db=audio.minimum_snr_db,
            acceptable_snr_db=audio.acceptable_snr_db,
            min_activity_ratio=audio.min_activity_ratio,
            min_contiguous_active_frames=audio.min_contiguous_active_frames,
            min_audio_quality=audio.min_audio_quality,
            min_speaker_confidence=audio.min_speaker_confidence,
        )
        self.result_queue: LatestOnlyQueue[RuntimeSnapshot] = LatestOnlyQueue()
        self._closed = False
        self._last_rendered_timestamp_ms = -1
        self._draw_rgb = (244, 231, 210)
        self._draw_brightness = 0.12
        self._transition_callback: str | None = None
        self._build()
        self._update_recognition_status("未接入")
        self._render_lights()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_results()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="EmoLight", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="SIMULATION  ·  演示事件，不代表真实情绪或身份识别", foreground="#9a5600").pack(anchor="w", pady=(0, 8))
        self.status = ttk.Label(outer, text="", wraplength=580)
        self.status.pack(anchor="w", pady=(0, 12))
        self.quality = ttk.Label(outer, text="当前音频质量：未采样")
        self.quality.pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(outer, height=112, background="#171a20", highlightthickness=0)
        self.canvas.pack(fill="x", pady=(0, 12))
        self.mode = ttk.Label(outer, text="灯效：中性待机 · 亮度 12%")
        self.mode.pack(anchor="w")
        self.event = ttk.Label(outer, text="尚未运行演示")
        self.event.pack(anchor="w", pady=(0, 16))

        ttk.Label(outer, text="情绪分数（仅模拟时显示模拟值）").pack(anchor="w")
        score_row = ttk.Frame(outer)
        score_row.pack(anchor="w", pady=(3, 10))
        self.score_vars = {}
        for emotion in Emotion:
            value = tk.StringVar(value=f"{emotion.value}: —")
            self.score_vars[emotion] = value
            ttk.Label(score_row, textvariable=value, width=16).pack(side="left")
        self.arousal = ttk.Label(outer, text="声音唤醒度：未配置")
        self.arousal.pack(anchor="w", pady=(0, 8))
        ttk.Label(outer, text="事件时间轴").pack(anchor="w")
        self.timeline = tk.Listbox(outer, height=4)
        self.timeline.pack(fill="x", pady=(3, 10))

        ttk.Label(outer, text="选择模拟情绪").pack(anchor="w")
        buttons = ttk.Frame(outer)
        buttons.pack(anchor="w", pady=6)
        labels = ((Emotion.NEUTRAL, "中性"), (Emotion.HAPPY, "愉快"), (Emotion.ANGRY, "愤怒"), (Emotion.SAD, "悲伤"))
        for emotion, text in labels:
            ttk.Button(buttons, text=text, command=lambda value=emotion: self._demo(value)).pack(side="left", padx=(0, 6))

        controls = ttk.Frame(outer)
        controls.pack(anchor="w", pady=8)
        ttk.Button(controls, text="启动/停止麦克风", command=self._toggle_microphone).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(controls, text="自动灯光", variable=self.automatic, command=self._toggle_auto).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(controls, text="夜间模式", variable=self.night, command=self._toggle_night).pack(side="left", padx=(0, 12))
        ttk.Button(controls, text="手动暖白", command=self._manual).pack(side="left")
        ttk.Label(outer, text="安全限制：无闪烁 · 亮度上限 40%（夜间 12%）· 缓慢过渡", foreground="#555").pack(anchor="w", pady=(14, 0))

    def _demo(self, emotion: Emotion) -> None:
        event = make_demo_event(emotion, int(time.time() * 1000))
        command = self.policy.evaluate_simulation(event, night_mode=self.night.get())
        self.event.configure(text=f"SIMULATION · {emotion.value} · 模拟置信度 95%")
        for candidate, value in self.score_vars.items():
            value.set(f"{candidate.value}: {0.95 if candidate is emotion else 0.00:.0%}")
        self.arousal.configure(text="声音唤醒度：模拟值 0.60")
        self.timeline.insert(0, f"{time.strftime('%H:%M:%S')}  SIMULATION  {emotion.value}")
        if self.timeline.size() > 12:
            self.timeline.delete(12, tk.END)
        if command is not None and self.automatic.get() and not self.manual_override:
            self._apply(command)
        elif not self.automatic.get() and not self.manual_override:
            self.mode.configure(text="灯效保持不变 · 自动模式已关闭")

    def _manual(self) -> None:
        command = self.policy.manual((255, 225, 185), night_mode=self.night.get())
        self.manual_override = True
        self.automatic.set(False)
        self._apply(command)
        self.event.configure(text="手动控制 · 暖白")

    def _toggle_auto(self) -> None:
        if self.automatic.get():
            self.manual_override = False

    def _toggle_night(self) -> None:
        current = self.controller.current
        if not self.night.get() or current is None:
            return
        limit = self.policy.config.night_max_brightness
        if current.brightness > limit:
            self._apply(LightCommand(
                rgb=current.rgb,
                brightness=limit,
                transition_ms=current.transition_ms,
                effect=current.effect,
                mode=current.mode,
                simulation=current.simulation,
            ))

    def _toggle_microphone(self) -> None:
        if self.mic_source is not None:
            self._mic_session_id += 1
            source, self.mic_source = self.mic_source, None
            source.stop()
            self._update_recognition_status("已停止")
            return
        source = MicrophoneAudioSource(
            sample_rate=self.runtime.sample_rate,
            frame_ms=self.config.audio.capture_frame_ms,
        )
        self._mic_session_id += 1
        session_id = self._mic_session_id
        self.mic_source = source
        try:
            source.start(lambda frame: self._process_audio_frame(frame, session_id=session_id))
        except RuntimeError as error:
            self.mic_source = None
            self._update_recognition_status(f"不可用：{error}")
            return
        self._update_recognition_status("已连接")

    def _update_recognition_status(self, microphone_status: str) -> None:
        if self.model_load_error:
            emotion_status = "INVALID（模型加载失败）"
        else:
            model_status = getattr(self.runtime.predictor, "model_status", ModelStatus.NOT_CONFIGURED)
            emotion_status = model_status.value
        if self.config.speaker_model_path:
            speaker_status = "NOT_CONFIGURED（声纹适配器尚未接入）"
        else:
            speaker_status = "NOT_CONFIGURED"
        self.status.configure(
            text=(
                f"目标身份：{speaker_status} · 情绪模型：{emotion_status} · 麦克风：{microphone_status}"
                " · 身份未验证时实时情绪事件拒识"
            )
        )

    def _process_audio_frame(self, frame, *, session_id: int | None = None) -> None:
        if (
            self._closed
            or self.mic_source is None
            or (session_id is not None and session_id != self._mic_session_id)
        ):
            return
        snapshot = self.runtime.feed_snapshot(frame.samples, timestamp_ms=frame.timestamp_ms)
        if snapshot is not None and not self._closed and self.mic_source is not None and (
            session_id is None or session_id == self._mic_session_id
        ):
            self.result_queue.publish(snapshot)

    def _poll_results(self) -> None:
        if self._closed:
            return
        snapshot = self.result_queue.take_latest()
        if snapshot is not None and snapshot.timestamp_ms > self._last_rendered_timestamp_ms:
            self._show_live_status(snapshot)
            self._last_rendered_timestamp_ms = snapshot.timestamp_ms
        self.root.after(50, self._poll_results)

    def _show_live_status(self, snapshot: RuntimeSnapshot) -> None:
        event = snapshot.event
        features = snapshot.acoustic_features
        dropped = self.mic_source.dropped_frames if self.mic_source is not None else 0
        self.quality.configure(
            text=(
                f"当前音频质量：{features.audio_quality:.0%} ({features.quality_status.value})"
                f"  |  活动占比：{features.activity_ratio:.0%}  |  状态：{event.status.value}  |  丢帧：{dropped}"
            )
        )
        self.event.configure(text=f"LIVE · {event.status.value} · 未输出情绪预测")
        for emotion, value in self.score_vars.items():
            value.set(f"{emotion.value}: —")
        self.arousal.configure(text="声音唤醒度：模型未配置")
        self.timeline.insert(0, f"{time.strftime('%H:%M:%S')}  LIVE  {event.status.value}")
        if self.timeline.size() > 12:
            self.timeline.delete(12, tk.END)

    def _close(self) -> None:
        self._closed = True
        self._mic_session_id += 1
        if self.mic_source is not None:
            source, self.mic_source = self.mic_source, None
            source.stop()
        self.root.destroy()

    def _apply(self, command: LightCommand) -> None:
        if self._transition_callback is not None:
            try:
                self.root.after_cancel(self._transition_callback)
            except tk.TclError:
                pass
            self._transition_callback = None
        previous_rgb, previous_brightness = self._draw_rgb, self._draw_brightness
        self.controller.apply(command)
        self.mode.configure(text=f"灯效：{command.mode} · RGB {command.rgb} · 亮度 {command.brightness:.0%} · 过渡 {command.transition_ms} ms")
        started = time.monotonic()
        duration = max(1.5, command.transition_ms / 1000.0)

        def frame() -> None:
            fraction = min(1.0, (time.monotonic() - started) / duration)
            self._draw_rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(previous_rgb, command.rgb))
            self._draw_brightness = previous_brightness + (command.brightness - previous_brightness) * fraction
            self._render_lights()
            if fraction < 1.0:
                self._transition_callback = self.root.after(33, frame)
            else:
                self._transition_callback = None

        frame()

    def _render_lights(self) -> None:
        self.canvas.delete("all")
        rgb = tuple(round(channel * self._draw_brightness) for channel in self._draw_rgb)
        color = "#%02x%02x%02x" % rgb
        width = max(self.canvas.winfo_width(), 580)
        spacing = width / 24
        for index in range(24):
            x = spacing * (index + 0.5)
            self.canvas.create_oval(x - 9, 42, x + 9, 60, fill=color, outline="")


def launch(config: AppConfig | LightingConfig | None = None) -> None:
    root = tk.Tk()
    EmotionSimulatorApp(root, config)
    root.mainloop()
