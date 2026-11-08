from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _build_backend(name: str, seed: int) -> tuple[str, Any]:
    if name not in {"auto", "lightgbm", "sklearn"}:
        raise ValueError(f"unknown detector backend: {name!r}")
    if name in {"auto", "lightgbm"}:
        try:
            import lightgbm as lgb
        except ImportError:
            if name == "lightgbm":
                raise
        else:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                verbose=-1,
            )
            return "lightgbm", model
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_features=0.8,
        early_stopping=False,
        random_state=seed,
    )
    return "sklearn_hist_gradient_boosting", model


class ReliabilityDetector:
    def __init__(
        self, seed: int = 0, backend: str = "auto", columns: list[int] | None = None
    ) -> None:
        self.seed = seed
        self.columns = columns
        self.backend_name, self.model = _build_backend(backend, seed)
        self.fitted = False

    def _select(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be a 2-D array")
        return features if self.columns is None else features[:, self.columns]

    def fit(self, features: np.ndarray, is_stale: np.ndarray) -> ReliabilityDetector:
        labels = np.asarray(is_stale).astype(int)
        if len(labels) != len(features):
            raise ValueError("features and labels must have the same length")
        if len(set(labels.tolist())) < 2:
            raise ValueError("training labels must contain both classes")
        self.model.fit(self._select(features), labels)
        self.fitted = True
        return self

    def stale_probability(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("detector is not fitted")
        selected = self._select(features)
        if len(selected) == 0:
            return np.zeros(0)
        return self.model.predict_proba(selected)[:, 1]

    def reliability(self, features: np.ndarray) -> np.ndarray:
        return 1.0 - self.stale_probability(features)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ReliabilityDetector:
        detector = joblib.load(path)
        if not isinstance(detector, ReliabilityDetector):
            raise TypeError("file does not contain a ReliabilityDetector")
        return detector
