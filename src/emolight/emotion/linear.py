"""JSON-loaded linear emotion inference implemented with NumPy only."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from emolight.emotion.predictor import EmotionPrediction, ModelStatus
from emolight.events import Emotion, SystemStatus
from emolight.features.emotion_v1 import (
    EMOTION_FEATURE_NAMES,
    EMOTION_FEATURE_SCHEMA_VERSION,
    SIMPLE_FEATURE_NAMES,
    SIMPLE_FEATURE_SCHEMA_VERSION,
    EmotionFeatures,
)


MODEL_FORMAT_VERSION = 1
MODEL_VERSION = "emolight-linear-numpy-v1"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class NumpyLinearEmotionPredictor:
    """Portable linear classifier; this module does not import scikit-learn."""

    def __init__(self) -> None:
        self._artifact: dict | None = None
        self._model_status = ModelStatus.NOT_CONFIGURED
        self._model_size_bytes = 0
        self._validation_record_id: str | None = None

    @property
    def model_status(self) -> ModelStatus:
        return self._model_status

    @property
    def model_size_bytes(self) -> int:
        return self._model_size_bytes

    @property
    def validation_record_id(self) -> str | None:
        return self._validation_record_id

    @property
    def metadata(self) -> dict:
        return dict(self._artifact or {})

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sample_rate: int | None = None,
        expected_frame_ms: float | None = None,
        expected_hop_ms: float | None = None,
        expected_activity_rms_threshold: float | None = None,
    ) -> "NumpyLinearEmotionPredictor":
        predictor = cls()
        model_path = Path(path)
        if not model_path.is_file():
            return predictor
        try:
            artifact = json.loads(model_path.read_text(encoding="utf-8"))
            predictor._validate_artifact(
                artifact,
                expected_sample_rate=expected_sample_rate,
                expected_frame_ms=expected_frame_ms,
                expected_hop_ms=expected_hop_ms,
                expected_activity_rms_threshold=expected_activity_rms_threshold,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            predictor._model_status = ModelStatus.INVALID
            return predictor
        predictor._artifact = artifact
        predictor._model_size_bytes = model_path.stat().st_size
        predictor._validation_record_id = sha256(
            json.dumps(artifact["validation"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        predictor._model_status = (
            ModelStatus.READY if artifact["calibration_status"] == "CALIBRATED" else ModelStatus.UNCALIBRATED
        )
        return predictor

    @staticmethod
    def _validate_artifact(
        artifact: dict,
        *,
        expected_sample_rate: int | None,
        expected_frame_ms: float | None,
        expected_hop_ms: float | None,
        expected_activity_rms_threshold: float | None,
    ) -> None:
        if not isinstance(artifact, dict) or artifact.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("unsupported JSON linear model format")
        if artifact.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported linear model version")
        profile = artifact.get("feature_profile")
        names, schema = (
            (EMOTION_FEATURE_NAMES, EMOTION_FEATURE_SCHEMA_VERSION)
            if profile == "full"
            else (SIMPLE_FEATURE_NAMES, SIMPLE_FEATURE_SCHEMA_VERSION)
            if profile == "simple"
            else ((), "")
        )
        if not names or artifact.get("feature_schema_version") != schema:
            raise ValueError("unsupported feature profile or schema")
        if tuple(artifact.get("feature_names", ())) != names:
            raise ValueError("model feature names do not match its schema")
        if expected_sample_rate is not None and artifact.get("sample_rate") != expected_sample_rate:
            raise ValueError("model sample rate mismatch")
        sample_rate = artifact.get("sample_rate")
        if type(sample_rate) is not int or sample_rate <= 1000:
            raise ValueError("invalid model sample rate")
        for key, expected in (
            ("frame_ms", expected_frame_ms),
            ("hop_ms", expected_hop_ms),
            ("activity_rms_threshold", expected_activity_rms_threshold),
        ):
            if expected is not None and artifact.get(key) != expected:
                raise ValueError(f"model {key} mismatch")
        if artifact.get("frame_ms", 0) <= 0 or artifact.get("hop_ms", 0) <= 0 or artifact["hop_ms"] > artifact["frame_ms"]:
            raise ValueError("invalid model frame/hop metadata")
        window_s = artifact.get("window_s")
        update_interval_s = artifact.get("update_interval_s")
        if (
            not isinstance(window_s, (int, float))
            or not np.isfinite(window_s)
            or window_s <= 0
            or not isinstance(update_interval_s, (int, float))
            or not np.isfinite(update_interval_s)
            or update_interval_s <= 0
            or update_interval_s > window_s
        ):
            raise ValueError("invalid model live window/update metadata")
        if artifact.get("activity_rms_threshold", -1) < 0:
            raise ValueError("invalid model activity threshold")
        if not np.isfinite(artifact["frame_ms"]) or not np.isfinite(artifact["hop_ms"]) or not np.isfinite(artifact["activity_rms_threshold"]):
            raise ValueError("model feature settings must be finite")
        classes = tuple(artifact.get("classes", ()))
        allowed = {emotion.value for emotion in Emotion}
        if len(classes) < 2 or len(set(classes)) != len(classes) or not set(classes).issubset(allowed):
            raise ValueError("invalid emotion class order")
        feature_count = len(names)
        class_count = len(classes)
        numeric_vectors = (
            "imputation_values", "standardization_mean", "standardization_scale",
            "intercepts", "calibration_slopes", "calibration_intercepts",
        )
        for key in numeric_vectors:
            values = np.asarray(artifact.get(key), dtype=np.float64)
            expected_length = class_count if key in ("intercepts", "calibration_slopes", "calibration_intercepts") else feature_count
            if values.shape != (expected_length,) or not np.isfinite(values).all():
                raise ValueError(f"invalid model parameter vector: {key}")
        if np.any(np.asarray(artifact["standardization_scale"], dtype=np.float64) <= 0.0):
            raise ValueError("standardization scales must be positive")
        weights = np.asarray(artifact.get("weights"), dtype=np.float64)
        if weights.shape != (class_count, feature_count) or not np.isfinite(weights).all():
            raise ValueError("invalid linear model weight matrix")
        threshold = artifact.get("rejection_threshold")
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("invalid validation-selected rejection threshold")
        validation = artifact.get("validation")
        if not isinstance(validation, dict) or int(validation.get("sample_count", 0)) <= 0:
            raise ValueError("independent validation record is required")
        training = artifact.get("training")
        if not isinstance(training, dict) or int(training.get("sample_count", 0)) <= 0:
            raise ValueError("training provenance is required")
        if not isinstance(artifact.get("dataset_split"), dict) or not artifact["dataset_split"].get("method"):
            raise ValueError("speaker-exclusive split provenance is required")
        if validation.get("threshold_selected_on") != "validation":
            raise ValueError("rejection threshold must be selected on validation data")
        threshold_method = validation.get("threshold_selection_method")
        if not isinstance(threshold_method, str) or "validation" not in threshold_method.casefold() or "test" in threshold_method.casefold():
            raise ValueError("validation-only threshold selection method is required")
        training_hash = artifact.get("training_manifest_sha256")
        validation_hash = validation.get("manifest_sha256")
        label_hash = artifact.get("label_mapping_sha256")
        if not all(_is_sha256(value) for value in (training_hash, validation_hash, label_hash)):
            raise ValueError("validation provenance and threshold method are required")
        if training_hash != validation_hash:
            raise ValueError("training and validation must derive from the same recorded manifest")
        if not isinstance(artifact.get("label_mapping"), dict):
            raise ValueError("label mapping provenance is required")
        training_library = artifact.get("training_library")
        if not isinstance(training_library, dict):
            raise ValueError("training library parameters are required")
        if (
            not training_library.get("scikit_learn_version")
            or training_library.get("classifier") != "LinearSVC"
            or training_library.get("dual") != "auto"
            or training_library.get("standardization") != "training split only"
            or training_library.get("imputation") != "per-feature training median; fail if a column has no observations"
        ):
            raise ValueError("unsupported or incomplete training metadata")
        classifier_c = training_library.get("C")
        if not isinstance(classifier_c, (int, float)) or not np.isfinite(classifier_c) or classifier_c <= 0.0:
            raise ValueError("invalid LinearSVC C metadata")
        train_counts = training.get("class_counts", {})
        validation_counts = validation.get("class_counts", {})
        if any(int(train_counts.get(label, 0)) <= 0 for label in classes):
            raise ValueError("training provenance lacks one or more model classes")
        if any(int(validation_counts.get(label, 0)) <= 0 for label in classes):
            raise ValueError("validation calibration provenance lacks one or more model classes")
        for key in ("brier_score", "ece"):
            metric = validation.get(key)
            if not isinstance(metric, (int, float)) or not np.isfinite(metric) or not 0.0 <= metric <= 1.0:
                raise ValueError(f"invalid validation calibration metric: {key}")
        if artifact.get("calibration_status") not in ("CALIBRATED", "UNCALIBRATED"):
            raise ValueError("unknown calibration status")
        if artifact.get("calibration_method") != "one-vs-rest sigmoid on independent validation split":
            raise ValueError("unexpected calibration method")

        quality_gates = artifact.get("quality_gates")
        required_gates = {
            "min_activity_ratio", "min_contiguous_active_frames", "clipping_limit",
            "minimum_snr_db", "acceptable_snr_db",
        }
        if not isinstance(quality_gates, dict) or not required_gates.issubset(quality_gates):
            raise ValueError("model quality gate settings are missing")

    def predict(self, features: EmotionFeatures, timestamp_ms: int = 0) -> EmotionPrediction:
        artifact = self._artifact
        if self._model_status is not ModelStatus.READY or artifact is None:
            rejection = SystemStatus.NOT_CONFIGURED if self._model_status is ModelStatus.NOT_CONFIGURED else SystemStatus.UNCERTAIN
            return EmotionPrediction(
                emotion=None,
                model_status=self._model_status,
                rejection_status=rejection,
                timestamp_ms=timestamp_ms,
                model_version=artifact.get("model_version") if artifact else None,
                calibration_status=artifact.get("calibration_status", "UNCALIBRATED") if artifact else "UNCALIBRATED",
            )
        if not self._features_match(features, artifact):
            return EmotionPrediction(
                emotion=None,
                model_status=ModelStatus.INVALID,
                rejection_status=SystemStatus.UNCERTAIN,
                timestamp_ms=timestamp_ms,
                model_version=artifact["model_version"],
                calibration_status=artifact["calibration_status"],
                decision_threshold=artifact["rejection_threshold"],
                validation_record_id=self._validation_record_id,
            )

        raw = np.asarray(features.values if artifact["feature_profile"] == "full" else features.simple_values, dtype=np.float64)
        imputation = np.asarray(artifact["imputation_values"], dtype=np.float64)
        if np.isinf(raw).any() or np.any(np.isnan(raw) & ~np.isnan(imputation)):
            raw = np.where(np.isfinite(raw), raw, imputation)
        else:
            raw = np.nan_to_num(raw, nan=0.0)
        scaled = (raw - np.asarray(artifact["standardization_mean"], dtype=np.float64)) / np.asarray(
            artifact["standardization_scale"], dtype=np.float64
        )
        margins = np.asarray(artifact["weights"], dtype=np.float64) @ scaled + np.asarray(artifact["intercepts"], dtype=np.float64)
        logits = np.clip(
            np.asarray(artifact["calibration_slopes"], dtype=np.float64) * margins
            + np.asarray(artifact["calibration_intercepts"], dtype=np.float64),
            -60.0,
            60.0,
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        total = float(np.sum(probabilities))
        if not np.isfinite(total) or total <= 0.0:
            return EmotionPrediction(
                emotion=None,
                model_status=ModelStatus.ERROR,
                rejection_status=SystemStatus.UNCERTAIN,
                timestamp_ms=timestamp_ms,
                model_version=artifact["model_version"],
                calibration_status=artifact["calibration_status"],
                decision_threshold=artifact["rejection_threshold"],
                validation_record_id=self._validation_record_id,
            )
        probabilities /= total
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        class_scores = {Emotion(label): float(score) for label, score in zip(artifact["classes"], probabilities)}
        rejection = SystemStatus.UNCERTAIN if confidence < artifact["rejection_threshold"] else None
        return EmotionPrediction(
            emotion=None if rejection else Emotion(artifact["classes"][index]),
            confidence=confidence,
            scores=class_scores,
            model_status=ModelStatus.READY,
            rejection_status=rejection,
            timestamp_ms=timestamp_ms,
            model_version=artifact["model_version"],
            calibration_status=artifact["calibration_status"],
            decision_threshold=float(artifact["rejection_threshold"]),
            validation_record_id=self._validation_record_id,
        )

    @staticmethod
    def _features_match(features: EmotionFeatures, artifact: dict) -> bool:
        if (
            features.sample_rate != artifact["sample_rate"]
            or features.frame_ms != artifact["frame_ms"]
            or features.hop_ms != artifact["hop_ms"]
            or features.activity_rms_threshold != artifact["activity_rms_threshold"]
        ):
            return False
        if artifact["feature_profile"] == "full":
            return (
                features.schema_version == artifact["feature_schema_version"]
                and features.feature_names == tuple(artifact["feature_names"])
            )
        return (
            features.simple_schema_version == artifact["feature_schema_version"]
            and features.simple_feature_names == tuple(artifact["feature_names"])
        )
