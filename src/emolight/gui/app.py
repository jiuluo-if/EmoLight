import time
import tkinter as tk
from tkinter import ttk

from emolight.audio.microphone import MicrophoneAudioSource
from emolight.config import LightingConfig
from emolight.demo import make_demo_event
from emolight.events import Emotion
from emolight.lighting.controller import SimulatorLightController
from emolight.lighting.policy import EmotionLightingPolicy, LightCommand
from emolight.runtime.pipeline import RealtimeFeatureRuntime


class EmotionSimulatorApp:
    def __init__(self, root: tk.Tk, config: LightingConfig | None = None) -> None:
        self.root = root
        self.root.title("EmoLight | 本地灯光模拟器")
        self.root.minsize(620, 470)
        self.policy = EmotionLightingPolicy(config or LightingConfig())
        self.controller = SimulatorLightController()
        self.automatic = tk.BooleanVar(value=True)
        self.night = tk.BooleanVar(value=False)
        self.manual_override = False
        self.mic_source: MicrophoneAudioSource | None = None
        self.runtime = RealtimeFeatureRuntime()
        self._draw_rgb = (244, 231, 210)
        self._draw_brightness = 0.12
        self._transition_callback: str | None = None
        self._build()
        self._render_lights()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="EmoLight", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="SIMULATION  ·  演示事件，不代表真实情绪或身份识别", foreground="#9a5600").pack(anchor="w", pady=(0, 8))
        self.status = ttk.Label(outer, text="真实识别：NOT_CONFIGURED  |  麦克风：未接入")
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
            self.mic_source.stop()
            self.mic_source = None
            self.status.configure(text="真实识别：NOT_CONFIGURED  |  麦克风：已停止")
            return
        source = MicrophoneAudioSource(sample_rate=self.runtime.sample_rate)
        try:
            source.start(self._process_audio_frame)
        except RuntimeError as error:
            self.status.configure(text=f"麦克风不可用：{error}")
            return
        self.mic_source = source
        self.status.configure(text="真实识别：NOT_CONFIGURED  |  麦克风：已连接，情绪自动灯光仍拒识")

    def _process_audio_frame(self, samples) -> None:
        event = self.runtime.feed(samples, timestamp_ms=round(time.monotonic() * 1000))
        if event is not None:
            try:
                self.root.after(0, lambda result=event: self._show_live_status(result))
            except tk.TclError:
                pass

    def _show_live_status(self, event) -> None:
        quality = self.runtime.latest_features.audio_quality if self.runtime.latest_features else 0.0
        dropped = self.mic_source.dropped_frames if self.mic_source is not None else 0
        self.quality.configure(text=f"当前音频质量：{quality:.0%}  |  状态：{event.status.value}  |  丢帧：{dropped}")
        self.event.configure(text=f"LIVE · {event.status.value} · 未输出情绪预测")
        for emotion, value in self.score_vars.items():
            value.set(f"{emotion.value}: —")
        self.arousal.configure(text="声音唤醒度：模型未配置")
        self.timeline.insert(0, f"{time.strftime('%H:%M:%S')}  LIVE  {event.status.value}")
        if self.timeline.size() > 12:
            self.timeline.delete(12, tk.END)

    def _close(self) -> None:
        if self.mic_source is not None:
            self.mic_source.stop()
            self.mic_source = None
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


def launch(config: LightingConfig | None = None) -> None:
    root = tk.Tk()
    EmotionSimulatorApp(root, config)
    root.mainloop()
