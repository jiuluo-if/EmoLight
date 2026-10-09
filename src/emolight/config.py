from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class LightingConfig:
    max_brightness: float = 0.40
    night_max_brightness: float = 0.12
    transition_ms: int = 3000
    allow_flicker: bool = False
    min_emotion_confidence: float = 0.70
    min_speaker_confidence: float = 0.80
    min_audio_quality: float = 0.50
    neutral_rgb: tuple[int, int, int] = (244, 231, 210)
    happy_rgb: tuple[int, int, int] = (255, 224, 158)
    angry_rgb: tuple[int, int, int] = (178, 205, 194)
    sad_rgb: tuple[int, int, int] = (255, 197, 120)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_brightness <= 1.0:
            raise ValueError("max_brightness must be in (0, 1]")
        if not 0.0 <= self.night_max_brightness <= self.max_brightness:
            raise ValueError("night_max_brightness must be between 0 and max_brightness")
        if self.transition_ms < 1500:
            raise ValueError("transition_ms must be at least 1500")
        if self.allow_flicker:
            raise ValueError("flicker effects are disabled for safety")
        for name in ("min_emotion_confidence", "min_speaker_confidence", "min_audio_quality"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("neutral_rgb", "happy_rgb", "angry_rgb", "sad_rgb"):
            try:
                color = tuple(getattr(self, name))
            except TypeError as error:
                raise ValueError(f"{name} must contain three integer RGB channels") from error
            if len(color) != 3 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in color):
                raise ValueError(f"{name} must contain three integer RGB channels in [0, 255]")
            object.__setattr__(self, name, color)


def load_lighting_config(path: str | Path) -> LightingConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("lighting"), dict):
        raise ValueError("configuration must contain a lighting object")
    allowed = set(LightingConfig.__dataclass_fields__)
    unknown = set(document["lighting"]) - allowed
    if unknown:
        raise ValueError(f"unknown lighting settings: {', '.join(sorted(unknown))}")
    values = document["lighting"]
    return LightingConfig(**values)
