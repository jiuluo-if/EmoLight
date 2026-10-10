"""CPU-only live or WAV-replay emotion predictions with an explicit experimental mode."""

import argparse
from collections import deque
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Sequence

import numpy as np

from emolight.audio.microphone import MicrophoneAudioSource
from emolight.audio.wav import load_wav
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.emotion.predictor import ModelStatus
from emolight.events import Emotion
from emolight.runtime.experimental import EmotionOnlyExperimentalRuntime


class LatencySamples:
    """Keep a rolling latency sample window while counting the full session."""

    def __init__(self, capacity: int = 4096) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("latency sample capacity must be a positive integer")
        self._values: deque[float] = deque(maxlen=capacity)
        self.sample_count = 0

    def add(self, value: float) -> None:
        measured = float(value)
        if not np.isfinite(measured) or measured < 0:
            raise ValueError("latency samples must be finite and non-negative")
        self._values.append(measured)
        self.sample_count += 1

    @property
    def retained_sample_count(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)


def _publish_latest(results: queue.Queue, value) -> bool:
    """Publish without waiting; return true when an older display update was replaced."""
    try:
        results.put_nowait(value)
        return False
    except queue.Full:
        try:
            results.get_nowait()
        except queue.Empty:
            pass
        try:
            results.put_nowait(value)
        except queue.Full:
            return True
        return True


def replay_wav(
    wav_path: str | Path,
    runtime: EmotionOnlyExperimentalRuntime,
    *,
    frame_ms: int = 20,
    pace: bool = False,
) -> tuple[list, dict]:
    """Feed a WAV through the same bounded 1.5 s/0.5 s path as microphone audio."""
    if type(frame_ms) is not int or frame_ms <= 0:
        raise ValueError("capture frame duration must be a positive integer")
    audio = load_wav(wav_path)
    if audio.sample_rate != runtime.sample_rate:
        raise ValueError(f"WAV sample rate mismatch: expected {runtime.sample_rate}, got {audio.sample_rate}")
    block_samples = max(1, round(audio.sample_rate * frame_ms / 1000.0))
    results = []
    frame_processing_ms = LatencySamples()
    result_processing_ms = LatencySamples()
    process = _try_process()
    cpu_before = process.cpu_times() if process else None
    peak_rss = process.memory_info().rss if process else None
    started = time.perf_counter()
    for start in range(0, audio.samples.size, block_samples):
        chunk = audio.samples[start:start + block_samples]
        end_sample = start + chunk.size
        timestamp_ms = round(end_sample * 1000 / audio.sample_rate)
        frame_started = time.perf_counter_ns()
        result = runtime.feed(chunk, timestamp_ms)
        elapsed_ms = (time.perf_counter_ns() - frame_started) / 1e6
        frame_processing_ms.add(elapsed_ms)
        if result is not None:
            results.append(result)
            result_processing_ms.add(elapsed_ms)
        if process:
            rss = process.memory_info().rss
            peak_rss = max(peak_rss or rss, rss)
        if pace:
            time.sleep(chunk.size / audio.sample_rate)
    wall_seconds = time.perf_counter() - started
    cpu_seconds = None
    if process and cpu_before:
        cpu_after = process.cpu_times()
        cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    return results, {
        "mode": "WAV_REPLAY_EMOTION_ONLY_EXPERIMENTAL",
        "identity_status": "NOT_EVALUATED",
        "audio_duration_s": audio.duration_s,
        "capture_frame_ms": frame_ms,
        "capture_frame_count": frame_processing_ms.sample_count,
        "window_updates": len(results),
        "processing_ms": {
            "per_capture_frame": _latency(frame_processing_ms),
            "per_window_update": _latency(result_processing_ms),
        },
        "process_cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "average_cpu_cores_used": cpu_seconds / wall_seconds if cpu_seconds is not None and wall_seconds else None,
        "peak_rss_bytes": peak_rss,
        "real_time_factor": audio.duration_s / wall_seconds if wall_seconds > 0 else None,
    }


def _try_process():
    try:
        import psutil

        return psutil.Process()
    except ImportError:
        return None


def _latency(values: LatencySamples) -> dict:
    retained = values.values
    median = float(np.median(retained)) if retained else None
    return {
        "sample_count": values.sample_count,
        "retained_sample_count": values.retained_sample_count,
        "median": median,
        "p50": median,
        "p95": float(np.quantile(retained, 0.95)) if retained else None,
    }


def _parameter_count(artifact: dict) -> int:
    keys = (
        "weights", "intercepts", "standardization_mean", "standardization_scale", "imputation_values",
        "calibration_slopes", "calibration_intercepts",
    )
    return sum(int(np.asarray(artifact[key]).size) for key in keys)


def _display_provenance(provenance):
    if isinstance(provenance, list):
        return [_display_provenance(item) for item in provenance]
    if not isinstance(provenance, dict):
        return provenance
    private_keys = {"split_speaker_ids", "speaker_ids_by_split", "test_speaker_ids"}
    return {
        key: _display_provenance(value)
        for key, value in provenance.items()
        if key not in private_keys
    }


def _result_payload(result) -> dict:
    return {
        "mode": result.mode,
        "identity_status": result.identity_status,
        "timestamp_ms": result.timestamp_ms,
        "emotion": result.emotion.value if result.emotion else None,
        "candidate_emotion": result.candidate_emotion.value if result.candidate_emotion else None,
        "scores": {emotion.value: float(result.scores.get(emotion, 0.0)) for emotion in Emotion},
        "confidence": result.confidence,
        "confidence_for": result.candidate_emotion.value if result.candidate_emotion else None,
        "rejection_reason": result.rejection_reason,
        "audio_activity": result.audio_active,
        "activity_ratio": result.activity_ratio,
        "audio_quality": result.audio_quality,
        "audio_quality_status": result.audio_quality_status,
        "calibration_status": result.calibration_status,
        "rejection_threshold": result.rejection_threshold,
        "model_version": result.model_version,
        "smoothing": "accepted-label hysteresis: switch after 2 of 3 recent accepted windows",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="live_emotion_only.py",
        description="Run explicitly experimental emotion-only predictions; never asserts target identity or emits lamp events.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--microphone", action="store_true", help="read 16 kHz mono audio from the default input device")
    source.add_argument("--wav", help="replay a local 16 kHz mono WAV through the same rolling-window runtime")
    parser.add_argument("--model", required=True, help="trained, validation-calibrated NumPy JSON model")
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--pace", action="store_true", help="pace WAV replay at its recorded duration")
    parser.add_argument("--duration-s", type=float, help="stop microphone mode after this many seconds")
    args = parser.parse_args(argv)
    try:
        predictor = NumpyLinearEmotionPredictor.load(args.model)
        if predictor.model_status is not ModelStatus.READY:
            parser.exit(2, f"live_emotion_only.py: model is {predictor.model_status.value}; live prediction is withheld\n")
        artifact = predictor.metadata
        expected_classes = {emotion.value for emotion in Emotion}
        if set(artifact.get("classes", ())) != expected_classes:
            parser.exit(2, "live_emotion_only.py: model must contain neutral/happy/angry/sad classes\n")
        runtime = EmotionOnlyExperimentalRuntime(predictor)
        print(json.dumps({
            "type": "session",
            "mode": "EMOTION_ONLY_EXPERIMENTAL",
            "identity_status": "NOT_EVALUATED",
            "target_conditioned_live": "NOT_CONFIGURED; use the existing speaker-gated runtime for target attribution",
            "sample_rate_hz": runtime.sample_rate,
            "window_s": runtime.window_s,
            "update_interval_s": runtime.update_interval_s,
            "model_profile": artifact["feature_profile"],
            "model_version": artifact["model_version"],
            "model_size_bytes": predictor.model_size_bytes,
            "parameter_count": _parameter_count(artifact),
            "dataset_provenance": _display_provenance(artifact.get("dataset_provenance")),
            "warning": "Predictions describe audio only; they are not attributed to a named person and cannot trigger real lighting.",
            "microphone_privacy": "Microphone audio is processed locally for acoustic analysis; raw audio and personal emotion history are not saved, and audio is not sent over a network. Obtain consent from everyone whose voice may be captured.",
        }, ensure_ascii=False, allow_nan=False))
        if args.wav:
            results, summary = replay_wav(args.wav, runtime, frame_ms=args.frame_ms, pace=args.pace)
            for result in results:
                print(json.dumps(_result_payload(result), ensure_ascii=False, allow_nan=False))
            print(json.dumps({"type": "summary", **summary}, ensure_ascii=False, allow_nan=False))
            return 0
        return _run_microphone(runtime, predictor, args.frame_ms, args.duration_s)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"live_emotion_only.py: operation failed ({type(error).__name__})", file=sys.stderr)
        return 2


def _run_microphone(runtime, predictor, frame_ms: int, duration_s: float | None) -> int:
    if duration_s is not None and (not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration must be finite and positive")
    source = MicrophoneAudioSource(sample_rate=runtime.sample_rate, frame_ms=frame_ms)
    lock = threading.Lock()
    display_results: queue.Queue[dict] = queue.Queue(maxsize=1)
    processing_ms = LatencySamples()
    window_processing_ms = LatencySamples()
    display_dropped_updates = 0
    peak_rss = None
    process = _try_process()
    cpu_before = process.cpu_times() if process else None
    started = time.perf_counter()

    def on_audio(frame) -> None:
        nonlocal peak_rss, display_dropped_updates
        began = time.perf_counter_ns()
        result = runtime.feed(frame.samples, frame.timestamp_ms)
        elapsed = (time.perf_counter_ns() - began) / 1e6
        with lock:
            processing_ms.add(elapsed)
            if process:
                rss = process.memory_info().rss
                peak_rss = max(peak_rss or rss, rss)
            if result is not None:
                window_processing_ms.add(elapsed)
                if _publish_latest(display_results, _result_payload(result)):
                    display_dropped_updates += 1

    try:
        source.start(on_audio)
        while duration_s is None or time.perf_counter() - started < duration_s:
            try:
                payload = display_results.get_nowait()
            except queue.Empty:
                pass
            else:
                print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if not source.worker_alive:
            runtime.clear_audio_cache()
    try:
        payload = display_results.get_nowait()
    except queue.Empty:
        pass
    else:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
    wall_seconds = time.perf_counter() - started
    cpu_seconds = None
    if process and cpu_before:
        cpu_after = process.cpu_times()
        cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    print(json.dumps({
        "type": "summary",
        "mode": "MICROPHONE_EMOTION_ONLY_EXPERIMENTAL",
        "identity_status": "NOT_EVALUATED",
        "audio_queue_dropped_frames": source.dropped_frames,
        "display_dropped_updates": display_dropped_updates,
        "processing_ms_per_capture_frame": _latency(processing_ms),
        "processing_ms_per_window_update": _latency(window_processing_ms),
        "process_cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "average_cpu_cores_used": cpu_seconds / wall_seconds if cpu_seconds is not None and wall_seconds else None,
        "peak_rss_bytes": peak_rss,
        "model_size_bytes": predictor.model_size_bytes,
        "parameter_count": _parameter_count(predictor.metadata),
    }, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
