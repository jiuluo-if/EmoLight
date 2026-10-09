import numpy as np

from emolight.audio.vad import EnergyVAD
from emolight.audio.wav import AudioBuffer
from emolight.events import SystemStatus
from emolight.speaker.verifier import UnconfiguredSpeakerVerifier


def test_energy_vad_marks_silence_and_active_frames():
    samples = np.concatenate((np.zeros(400, dtype=np.float32), np.full(400, 0.1, dtype=np.float32)))
    frames = EnergyVAD(frame_ms=25, threshold_rms=0.02).process(AudioBuffer(samples, 16000))

    assert len(frames) == 2
    assert frames[0].status is SystemStatus.SILENCE
    assert frames[1].status is SystemStatus.UNCERTAIN
    assert frames[1].energy_rms > 0.02


def test_speaker_verifier_never_claims_unconfigured_identity():
    result = UnconfiguredSpeakerVerifier().verify(AudioBuffer(np.ones(100, dtype=np.float32), 16000))

    assert result.status is SystemStatus.NOT_CONFIGURED
    assert result.confidence == 0.0
