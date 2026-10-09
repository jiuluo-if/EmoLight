import json

from emolight.config import AppConfig, load_app_config, load_lighting_config
from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus
from emolight.lighting.policy import EmotionLightingPolicy


def test_lighting_configuration_is_loaded_from_json_and_applied(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"lighting": {"max_brightness": 0.18, "night_max_brightness": 0.08, "transition_ms": 2200}}), encoding="utf-8")
    config = load_lighting_config(path)
    event = EmotionEvent(Emotion.HAPPY, SystemStatus.TARGET_ACTIVE, 0.9, 0.9, 0.9)

    command = EmotionLightingPolicy(config).evaluate(event)
    assert command is not None
    assert command.brightness == 0.18
    assert command.transition_ms == 2200


def test_user_can_customize_emotion_colors(tmp_path):
    path = tmp_path / "colors.json"
    path.write_text(json.dumps({"lighting": {"happy_rgb": [10, 20, 30]}}), encoding="utf-8")
    policy = EmotionLightingPolicy(load_lighting_config(path))
    event = EmotionEvent(
        Emotion.HAPPY,
        SystemStatus.TARGET_ACTIVE,
        0.9,
        0.9,
        0.9,
        source=EventSource.SIMULATION,
    )

    command = policy.evaluate_simulation(event)
    assert command is not None
    assert command.rgb == (10, 20, 30)


def test_app_config_unifies_audio_model_and_lighting_values(tmp_path):
    path = tmp_path / "app.json"
    path.write_text(json.dumps({
        "audio": {
            "sample_rate_hz": 8000,
            "channels": 1,
            "capture_frame_ms": 20,
            "frame_ms": 25,
            "hop_ms": 10,
            "emotion_window_s": 2.0,
            "update_interval_s": 0.4,
            "min_activity_ratio": 0.3,
        },
        "lighting": {"max_brightness": 0.2},
        "models": {"emotion": "models/local.joblib", "speaker": None},
    }), encoding="utf-8")

    config = load_app_config(path)

    assert isinstance(config, AppConfig)
    assert config.audio.sample_rate_hz == 8000
    assert config.audio.capture_frame_ms == 20
    assert config.audio.window_s == 2.0
    assert config.lighting.max_brightness == 0.2
    assert config.emotion_model_path == str((tmp_path / "models" / "local.joblib").resolve())


def test_app_config_rejects_audio_parameters_the_runtime_cannot_use(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"audio": {"frame_ms": 10, "hop_ms": 20}}), encoding="utf-8")

    try:
        load_app_config(path)
    except ValueError as error:
        assert "hop_ms" in str(error)
    else:
        raise AssertionError("hop longer than frame must be rejected")


def test_app_config_rejects_nonfinite_audio_parameters(tmp_path):
    for name, invalid_value in (
        ("frame_ms", float("nan")),
        ("hop_ms", float("inf")),
        ("window_s", float("inf")),
        ("activity_rms_threshold", float("nan")),
        ("minimum_snr_db", float("-inf")),
    ):
        path = tmp_path / f"bad-{name}.json"
        path.write_text(json.dumps({"audio": {name: invalid_value}}), encoding="utf-8")
        try:
            load_app_config(path)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError(f"non-finite {name} must be rejected")
