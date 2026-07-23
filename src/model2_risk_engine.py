from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from risk_engine import RiskFeatures, calculate_risk_from_features, risk_category


FEATURE_NAMES = [
    "ttc_inv",
    "time_to_impact_inv",
    "lane_distance",
    "relative_distance",
    "bbox_growth_rate",
    "bbox_width_growth_rate",
    "bbox_area_growth_rate_norm",
    "speed_px_s",
    "horizontal_speed_px_s_abs",
    "vertical_speed_px_s",
    "acceleration_px_s2",
    "trajectory_intersection",
    "impact_zone_intersection",
    "in_ego_corridor",
    "in_impact_zone",
    "near_enough",
    "moving_toward_ego_center",
    "approaching_camera",
    "collision_candidate",
]


@dataclass(frozen=True)
class Model2Prediction:
    score: float
    status: str
    color: tuple[int, int, int]
    source: str
    reasons: list[str]


class Model2RiskPredictor:
    def __init__(
        self,
        model_path: Path | None,
        frame_width: int,
        frame_height: int,
        fallback_to_rules: bool = True,
    ) -> None:
        self.frame_area = max(frame_width * frame_height, 1)
        self.fallback_to_rules = fallback_to_rules
        self.model_path = model_path
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        if model_path is not None and model_path.exists():
            data = np.load(model_path, allow_pickle=False)
            self.weights = data["weights"].astype(float)
            self.bias = float(data["bias"])
            self.mean = data["mean"].astype(float)
            self.std = data["std"].astype(float)
        elif model_path is not None and not fallback_to_rules:
            raise FileNotFoundError(f"Risk model tidak ditemukan: {model_path}")

    @property
    def has_ml_model(self) -> bool:
        return self.weights is not None and self.mean is not None and self.std is not None

    def feature_vector(self, features: RiskFeatures) -> np.ndarray:
        ttc_inv = 0.0 if features.ttc is None else 1.0 / max(features.ttc, 1e-3)
        impact_inv = (
            0.0
            if features.time_to_impact_zone is None
            else 1.0 / max(features.time_to_impact_zone, 1e-3)
        )
        return np.array(
            [
                ttc_inv,
                impact_inv,
                features.lane_distance,
                features.relative_distance,
                features.bbox_growth_rate,
                features.bbox_width_growth_rate,
                features.bbox_area_growth_rate / self.frame_area,
                features.speed_px_s,
                abs(features.horizontal_speed_px_s),
                features.vertical_speed_px_s,
                features.acceleration_px_s2,
                float(features.trajectory_intersection),
                float(features.impact_zone_intersection),
                float(features.in_ego_corridor),
                float(features.in_impact_zone),
                float(features.near_enough),
                float(features.moving_toward_ego_center),
                float(features.approaching_camera),
                float(features.collision_candidate),
            ],
            dtype=float,
        )

    def predict(
        self,
        features: RiskFeatures,
        bbox_height: float,
        frame_height: int,
    ) -> Model2Prediction:
        rule_score, reasons = calculate_risk_from_features(
            features=features,
            bbox_height=bbox_height,
            frame_height=frame_height,
        )

        if not self.has_ml_model:
            status, color = risk_category(rule_score)
            return Model2Prediction(
                score=rule_score,
                status=status,
                color=color,
                source="rule_fallback",
                reasons=["Model ML belum ada, memakai rule fallback"] + reasons,
            )

        assert self.weights is not None
        assert self.mean is not None
        assert self.std is not None

        vector = self.feature_vector(features)
        normalized = (vector - self.mean) / np.maximum(self.std, 1e-6)
        logit = float(np.dot(normalized, self.weights) + self.bias)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        ml_score = float(np.clip(probability * 100.0, 0.0, 100.0))

        status, color = risk_category(ml_score)
        return Model2Prediction(
            score=ml_score,
            status=status,
            color=color,
            source="ml",
            reasons=[f"ML probability={probability:.3f}"] + reasons,
        )
