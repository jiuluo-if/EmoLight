from dataclasses import dataclass
from typing import Protocol

from emolight.audio.wav import AudioBuffer
from emolight.events import SystemStatus


@dataclass(frozen=True)
class SpeakerVerification:
    status: SystemStatus
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class SpeakerVerifier(Protocol):
    def verify(self, audio: AudioBuffer) -> SpeakerVerification: ...


class UnconfiguredSpeakerVerifier:
    """Fail-closed identity adapter until a verified speaker model is supplied."""

    def verify(self, audio: AudioBuffer) -> SpeakerVerification:
        return SpeakerVerification(status=SystemStatus.NOT_CONFIGURED, confidence=0.0)
