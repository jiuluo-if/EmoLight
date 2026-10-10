"""Public compatibility import for the versioned 24-D nonsemantic prosody schema."""

from emolight.features.emotion_v1 import (
    EMOTION_FEATURE_NAMES,
    EMOTION_FEATURE_SCHEMA_VERSION,
    EmotionFeatures,
    extract_emotion_features,
)


PROSODY_SCHEMA_VERSION = EMOTION_FEATURE_SCHEMA_VERSION
PROSODY_FEATURE_NAMES = EMOTION_FEATURE_NAMES
ProsodyFeatures = EmotionFeatures
extract_prosody = extract_emotion_features

__all__ = [
    "PROSODY_SCHEMA_VERSION",
    "PROSODY_FEATURE_NAMES",
    "ProsodyFeatures",
    "extract_prosody",
]
