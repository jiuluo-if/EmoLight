from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np

from emolight.emotion.predictor import EmotionPrediction, ModelStatus
from emolight.events import Emotion, SystemStatus
from emolight.features.prosody import (
    PROSODY_FEATURE_NAMES,
    PROSODY_SCHEMA_VERSION,
    ProsodyFeatures,
)


MODEL_FORMAT_VERSION = 1
MODEL_VERSION = "emolight-calibrated-linear-svm-v1"


def fit_calibrated_svm(
    feature_vectors: np.ndarray,
    labels: Iterable[str | Emotion],
    groups: Iterable[str | int],
    *,
    seed: int = 42,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    activity_rms_threshold: float = 0.01,
    split_metadata: Mapping[str, object] | None = None,
) -> dict:
    """Fit scaling, LinearSVC, and sigmoid calibration on disjoint speaker groups."""
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ImportError as error:
        raise RuntimeError("SVM training requires the optional 'ml' dependencies") from error

    x = np.asarray(feature_vectors, dtype=np.float64)
    y_raw = np.asarray([label.value if isinstance(label, Emotion) else str(label) for label in labels])
    group_ids = np.asarray([str(group) for group in groups])
    if x.ndim != 2 or x.shape[1] != len(PROSODY_FEATURE_NAMES):
        raise ValueError(f"feature matrix must have shape (n, {len(PROSODY_FEATURE_NAMES)})")
    if x.shape[0] == 0 or y_raw.shape != (x.shape[0],) or group_ids.shape != y_raw.shape:
        raise ValueError("features, labels, and groups must have the same non-zero row count")
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains non-finite values")
    if not np.isfinite(activity_rms_threshold) or activity_rms_threshold < 0.0:
        raise ValueError("activity_rms_threshold must be finite and non-negative")
    allowed = {emotion.value for emotion in Emotion}
    if not set(y_raw).issubset(allowed):
        raise ValueError("labels must be neutral, happy, angry, or sad")
    classes = tuple(sorted(np.unique(y_raw).tolist()))
    if len(classes) < 2:
        raise ValueError("at least two emotion classes are required")
    class_group_counts = {
        label: len(set(group_ids[y_raw == label].tolist())) for label in classes
    }
    n_splits = min(3, min(class_group_counts.values()))
    if n_splits < 2:
        raise ValueError("each emotion needs at least two independent speaker groups for calibration")

    cv_splits = None
    group_audit = None
    for attempt in range(32):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        candidate = list(splitter.split(x, y_raw, groups=group_ids))
        all_classes = set(classes)
        valid = all(
            set(y_raw[train_indices]) == all_classes
            and set(y_raw[calibration_indices]) == all_classes
            and set(group_ids[train_indices]).isdisjoint(group_ids[calibration_indices])
            for train_indices, calibration_indices in candidate
        )
        if valid:
            cv_splits = candidate
            group_audit = [
                (tuple(sorted(set(group_ids[train_indices]))), tuple(sorted(set(group_ids[calibration_indices]))))
                for train_indices, calibration_indices in candidate
            ]
            break
    if cv_splits is None or group_audit is None:
        raise ValueError("cannot create class-complete calibration folds without speaker leakage")

    estimator = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=seed, dual="auto"),
    )
    classifier = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=cv_splits,
        ensemble=True,
    )
    classifier.fit(x, y_raw)
    model_classes = tuple(str(label) for label in classifier.classes_)
    if set(model_classes) != set(classes):
        raise RuntimeError("calibrated classifier changed the trained class set")

    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "feature_schema_version": PROSODY_SCHEMA_VERSION,
        "feature_names": list(PROSODY_FEATURE_NAMES),
        "feature_count": len(PROSODY_FEATURE_NAMES),
        "sample_rate": int(sample_rate),
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
        "activity_rms_threshold": float(activity_rms_threshold),
        "classes": list(model_classes),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn_version,
        "record_count": int(x.shape[0]),
        "speaker_group_count": int(len(set(group_ids))),
        "class_group_counts": class_group_counts,
        "calibration_method": "sigmoid",
        "calibration_splitter": "StratifiedGroupKFold",
        "calibration_fold_count": n_splits,
        "calibration_fold_group_counts": [
            {"fit": len(fit_groups), "calibration": len(calibration_groups)}
            for fit_groups, calibration_groups in group_audit
        ],
        "dataset_split": dict(split_metadata or {}),
    }
    return {
        "metadata": metadata,
        "classifier": classifier,
        # Ephemeral audit only. save_model_artifact never serializes group identifiers.
        "_calibration_group_audit": group_audit,
    }


def save_model_artifact(path: str | Path, artifact: Mapping[str, object]) -> int:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("model serialization requires the optional 'ml' dependencies") from error
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in artifact.items() if not key.startswith("_")}
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(serializable, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target.stat().st_size


class SklearnEmotionPredictor:
    """Local calibrated baseline. Load only model files produced from trusted local training."""

    def __init__(self) -> None:
        self._classifier = None
        self._metadata: dict | None = None
        self._load_status = ModelStatus.NOT_CONFIGURED
        self._model_size_bytes = 0

    @property
    def model_status(self) -> ModelStatus:
        return self._load_status

    @property
    def metadata(self) -> dict:
        return dict(self._metadata or {})

    @property
    def model_size_bytes(self) -> int:
        return self._model_size_bytes

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sample_rate: int | None = None,
        expected_frame_ms: float | None = None,
        expected_hop_ms: float | None = None,
        expected_activity_rms_threshold: float | None = None,
    ) -> "SklearnEmotionPredictor":
        predictor = cls()
        model_path = Path(path)
        if not model_path.is_file():
            return predictor
        try:
            import joblib

            artifact = joblib.load(model_path)
            metadata = artifact["metadata"]
            classifier = artifact["classifier"]
            predictor._validate_artifact(
                metadata,
                classifier,
                expected_sample_rate,
                expected_frame_ms,
                expected_hop_ms,
                expected_activity_rms_threshold,
            )
        except Exception:
            predictor._load_status = ModelStatus.INVALID
            return predictor
        predictor._classifier = classifier
        predictor._metadata = metadata
        predictor._load_status = ModelStatus.READY
        predictor._model_size_bytes = model_path.stat().st_size
        return predictor

    @staticmethod
    def _validate_artifact(
        metadata: dict,
        classifier,
        expected_sample_rate: int | None,
        expected_frame_ms: float | None,
        expected_hop_ms: float | None,
        expected_activity_rms_threshold: float | None,
    ) -> None:
        if not isinstance(metadata, dict) or metadata.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("unsupported model format")
        if metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported model version")
        if metadata.get("feature_schema_version") != PROSODY_SCHEMA_VERSION:
            raise ValueError("feature schema mismatch")
        if tuple(metadata.get("feature_names", ())) != PROSODY_FEATURE_NAMES:
            raise ValueError("feature names mismatch")
        if metadata.get("feature_count") != len(PROSODY_FEATURE_NAMES):
            raise ValueError("feature count mismatch")
        frame_ms = metadata.get("frame_ms")
        hop_ms = metadata.get("hop_ms")
        activity_threshold = metadata.get("activity_rms_threshold")
        if (
            not isinstance(frame_ms, (int, float))
            or not isinstance(hop_ms, (int, float))
            or not np.isfinite(frame_ms)
            or not np.isfinite(hop_ms)
            or frame_ms <= 0
            or hop_ms <= 0
            or hop_ms > frame_ms
        ):
            raise ValueError("invalid model frame/hop metadata")
        if (
            not isinstance(activity_threshold, (int, float))
            or not np.isfinite(activity_threshold)
            or activity_threshold < 0
        ):
            raise ValueError("invalid model activity threshold metadata")
        model_classes = tuple(metadata.get("classes", ()))
        allowed = {emotion.value for emotion in Emotion}
        if len(model_classes) < 2 or len(set(model_classes)) != len(model_classes) or not set(model_classes).issubset(allowed):
            raise ValueError("invalid model class order")
        if tuple(str(label) for label in classifier.classes_) != model_classes:
            raise ValueError("classifier class order does not match model metadata")
        if expected_sample_rate is not None and metadata.get("sample_rate") != expected_sample_rate:
            raise ValueError("model sampling rate mismatch")
        if expected_frame_ms is not None and metadata.get("frame_ms") != expected_frame_ms:
            raise ValueError("model analysis frame mismatch")
        if expected_hop_ms is not None and metadata.get("hop_ms") != expected_hop_ms:
            raise ValueError("model analysis hop mismatch")
        if expected_activity_rms_threshold is not None and metadata.get("activity_rms_threshold") != expected_activity_rms_threshold:
            raise ValueError("model activity threshold mismatch")
        if not hasattr(classifier, "predict_proba"):
            raise ValueError("model has no calibrated probability output")

    def predict(self, features: ProsodyFeatures, timestamp_ms: int = 0) -> EmotionPrediction:
        if self._load_status is not ModelStatus.READY or self._classifier is None or self._metadata is None:
            return EmotionPrediction(
                emotion=None,
                model_status=self._load_status,
                rejection_status=SystemStatus.NOT_CONFIGURED if self._load_status is ModelStatus.NOT_CONFIGURED else SystemStatus.UNCERTAIN,
                timestamp_ms=timestamp_ms,
            )
        if (
            features.schema_version != self._metadata["feature_schema_version"]
            or features.feature_names != tuple(self._metadata["feature_names"])
            or features.sample_rate != self._metadata["sample_rate"]
            or features.frame_ms != self._metadata["frame_ms"]
            or features.hop_ms != self._metadata["hop_ms"]
            or features.activity_rms_threshold != self._metadata["activity_rms_threshold"]
        ):
            return EmotionPrediction(None, model_status=ModelStatus.INVALID, timestamp_ms=timestamp_ms)
        try:
            probabilities = np.asarray(self._classifier.predict_proba(features.values.reshape(1, -1))[0], dtype=np.float64)
            if probabilities.shape != (len(self._metadata["classes"]),) or not np.isfinite(probabilities).all():
                raise ValueError("model returned invalid probabilities")
            index = int(np.argmax(probabilities))
            classes = self._metadata["classes"]
            emotion = Emotion(classes[index])
            scores = {Emotion(label): float(probability) for label, probability in zip(classes, probabilities)}
            return EmotionPrediction(
                emotion=emotion,
                confidence=float(probabilities[index]),
                scores=scores,
                model_status=ModelStatus.READY,
                timestamp_ms=timestamp_ms,
            )
        except Exception:
            return EmotionPrediction(
                None,
                model_status=ModelStatus.ERROR,
                rejection_status=SystemStatus.UNCERTAIN,
                timestamp_ms=timestamp_ms,
            )
