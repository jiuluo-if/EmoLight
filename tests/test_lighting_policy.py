from emolight.config import LightingConfig
from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus
from emolight.lighting.policy import EmotionLightingPolicy


def event(**overrides):
    values = dict(
        emotion=Emotion.HAPPY,
        status=SystemStatus.TARGET_ACTIVE,
        emotion_confidence=0.95,
        speaker_confidence=0.95,
        audio_quality=0.95,
        timestamp_ms=10,
    )
    values.update(overrides)
    return EmotionEvent(**values)


def test_live_policy_rejects_unverified_identity_quality_and_model_state():
    policy = EmotionLightingPolicy()

    assert policy.evaluate(event(status=SystemStatus.NON_TARGET)) is None
    assert policy.evaluate(event(speaker_confidence=0.1)) is None
    assert policy.evaluate(event(audio_quality=0.1)) is None
    assert policy.evaluate(EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED)) is None


def test_simulated_events_only_produce_simulation_preview():
    policy = EmotionLightingPolicy()
    simulated = event(source=EventSource.SIMULATION)

    assert policy.evaluate(simulated) is None
    preview = policy.evaluate_simulation(simulated)
    assert preview is not None and preview.simulation


def test_commands_are_bounded_and_manual_or_disabled_mode_wins():
    policy = EmotionLightingPolicy(LightingConfig())
    command = policy.evaluate(event(), night_mode=True)

    assert command is not None
    assert command.brightness <= 0.12
    assert command.transition_ms >= 1500
    assert command.effect == "steady"
    assert policy.evaluate(event(), automatic_enabled=False) is None
    assert policy.manual((20, 30, 40)).mode == "manual"


def test_emotions_map_to_non_flashing_warm_or_low_stimulation_colors():
    policy = EmotionLightingPolicy()
    happy = policy.evaluate(event(emotion=Emotion.HAPPY))
    angry = policy.evaluate(event(emotion=Emotion.ANGRY))
    sad = policy.evaluate(event(emotion=Emotion.SAD))

    assert happy and happy.rgb[0] > happy.rgb[2]
    assert angry and max(angry.rgb) < 230
    assert sad and sad.rgb[0] > sad.rgb[2]
    assert {happy.effect, angry.effect, sad.effect} == {"steady"}
