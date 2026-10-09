from dataclasses import dataclass

from emolight.config import LightingConfig
from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus


@dataclass(frozen=True)
class LightCommand:
    rgb: tuple[int, int, int]
    brightness: float
    transition_ms: int
    effect: str = "steady"
    mode: str = "automatic"
    simulation: bool = False


_BRIGHTNESS: dict[Emotion, float] = {
    Emotion.NEUTRAL: 0.24,
    Emotion.HAPPY: 0.34,
    Emotion.ANGRY: 0.20,
    Emotion.SAD: 0.22,
}


class EmotionLightingPolicy:
    """Convert trusted events to bounded commands; reject unsafe live events."""

    def __init__(self, config: LightingConfig | None = None) -> None:
        self.config = config or LightingConfig()

    def evaluate(
        self,
        event: EmotionEvent,
        *,
        automatic_enabled: bool = True,
        night_mode: bool = False,
    ) -> LightCommand | None:
        if not automatic_enabled or event.source is not EventSource.LIVE:
            return None
        if not self._is_trusted(event):
            return None
        return self._command(event.emotion, night_mode=night_mode)

    def evaluate_simulation(
        self,
        event: EmotionEvent,
        *,
        night_mode: bool = False,
    ) -> LightCommand | None:
        if event.source is not EventSource.SIMULATION or event.emotion is None:
            return None
        if event.status is not SystemStatus.TARGET_ACTIVE:
            return None
        return self._command(event.emotion, night_mode=night_mode, simulation=True)

    def manual(self, rgb: tuple[int, int, int], *, night_mode: bool = False) -> LightCommand:
        if len(rgb) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in rgb):
            raise ValueError("rgb must contain three integers in [0, 255]")
        brightness = min(0.35, self._brightness_limit(night_mode))
        return LightCommand(rgb=rgb, brightness=brightness, transition_ms=self.config.transition_ms, mode="manual")

    def _is_trusted(self, event: EmotionEvent) -> bool:
        return (
            event.emotion is not None
            and event.status is SystemStatus.TARGET_ACTIVE
            and event.emotion_confidence >= self.config.min_emotion_confidence
            and event.speaker_confidence >= self.config.min_speaker_confidence
            and event.audio_quality >= self.config.min_audio_quality
        )

    def _brightness_limit(self, night_mode: bool) -> float:
        return self.config.night_max_brightness if night_mode else self.config.max_brightness

    def _command(self, emotion: Emotion | None, *, night_mode: bool, simulation: bool = False) -> LightCommand | None:
        if emotion is None:
            return None
        rgb = getattr(self.config, f"{emotion.value}_rgb")
        requested = _BRIGHTNESS[emotion]
        brightness = min(requested, self._brightness_limit(night_mode))
        return LightCommand(
            rgb=rgb,
            brightness=brightness,
            transition_ms=max(1500, self.config.transition_ms),
            simulation=simulation,
        )
