from dataclasses import dataclass
import json
import math
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
        for name in ("max_brightness", "night_max_brightness", "min_emotion_confidence", "min_speaker_confidence", "min_audio_quality"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if type(self.transition_ms) is not int:
            raise ValueError("transition_ms must be an integer")
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


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    capture_frame_ms: int = 20
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    window_s: float = 1.5
    update_interval_s: float = 0.5
    activity_rms_threshold: float = 0.01
    min_activity_ratio: float = 0.20
    min_contiguous_active_frames: int = 3
    clipping_limit: float = 0.05
    minimum_snr_db: float = 5.0
    acceptable_snr_db: float = 10.0
    min_audio_quality: float = 0.50
    min_speaker_confidence: float = 0.80

    def __post_init__(self) -> None:
        integer_fields = ("sample_rate_hz", "channels", "capture_frame_ms", "min_contiguous_active_frames")
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        numeric_fields = (
            "frame_ms", "hop_ms", "window_s", "update_interval_s", "activity_rms_threshold",
            "min_activity_ratio", "clipping_limit", "minimum_snr_db", "acceptable_snr_db",
            "min_audio_quality", "min_speaker_confidence",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.sample_rate_hz <= 1000 or self.channels != 1:
            raise ValueError("audio must use mono and a sample rate above the configured F0 Nyquist limit")
        if self.capture_frame_ms <= 0 or self.frame_ms <= 0 or self.hop_ms <= 0 or self.hop_ms > self.frame_ms:
            raise ValueError("capture/frame durations must be positive and hop_ms must not exceed frame_ms")
        if self.window_s <= 0 or self.update_interval_s <= 0:
            raise ValueError("window_s and update_interval_s must be positive")
        if self.activity_rms_threshold < 0 or self.min_contiguous_active_frames < 1:
            raise ValueError("VAD threshold must be non-negative and contiguous active-frame count positive")
        if not 0.0 <= self.min_activity_ratio <= 1.0:
            raise ValueError("min_activity_ratio must be between 0 and 1")
        if not 0.0 <= self.clipping_limit <= 1.0 or self.minimum_snr_db >= self.acceptable_snr_db:
            raise ValueError("clipping and SNR thresholds are invalid")
        if not 0.0 <= self.min_audio_quality <= 1.0 or not 0.0 <= self.min_speaker_confidence <= 1.0:
            raise ValueError("quality and speaker thresholds must be between 0 and 1")


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = AudioConfig()
    lighting: LightingConfig = LightingConfig()
    emotion_model_path: str | None = None
    speaker_model_path: str | None = None


def load_lighting_config(path: str | Path) -> LightingConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("lighting"), dict):
        raise ValueError("configuration must contain a lighting object")
    allowed = set(LightingConfig.__dataclass_fields__)
    unknown = set(document["lighting"]) - allowed
    if unknown:
        raise ValueError(f"unknown lighting settings: {', '.join(sorted(unknown))}")
    return _lighting_config_from_mapping(document["lighting"])


def _lighting_config_from_mapping(values: dict) -> LightingConfig:
    allowed = set(LightingConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown lighting settings: {', '.join(sorted(unknown))}")
    return LightingConfig(**values)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("configuration root must be an object")
    audio_values = document.get("audio", {})
    lighting_values = document.get("lighting", {})
    models = document.get("models", {})
    if not isinstance(audio_values, dict) or not isinstance(lighting_values, dict) or not isinstance(models, dict):
        raise ValueError("audio, lighting, and models configuration sections must be objects")
    if "emotion_window_s" in audio_values:
        audio_values = dict(audio_values)
        audio_values["window_s"] = audio_values.pop("emotion_window_s")
    audio_allowed = set(AudioConfig.__dataclass_fields__)
    audio_unknown = set(audio_values) - audio_allowed
    if audio_unknown:
        raise ValueError(f"unknown audio settings: {', '.join(sorted(audio_unknown))}")
    model_unknown = set(models) - {"emotion", "speaker"}
    if model_unknown:
        raise ValueError(f"unknown model settings: {', '.join(sorted(model_unknown))}")

    def resolve_model(value, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"models.{name} must be a non-empty path or null")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return str(candidate.resolve())

    return AppConfig(
        audio=AudioConfig(**audio_values),
        lighting=_lighting_config_from_mapping(lighting_values),
        emotion_model_path=resolve_model(models.get("emotion"), "emotion"),
        speaker_model_path=resolve_model(models.get("speaker"), "speaker"),
    )
