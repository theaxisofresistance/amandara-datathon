from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from model2_risk_engine import FEATURE_NAMES


POSITIVE_STATUSES = {"DANGER", "HIGH RISK", "RISK", "NEAR-MISS", "NEARMISS", "1", "TRUE"}
NEGATIVE_STATUSES = {"SAFE", "WARNING", "0", "FALSE", "NORMAL"}


def parse_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_label(row: dict[str, str]) -> int | None:
    for name in ("label", "risk_label", "target", "collision", "is_collision"):
        if name in row:
            value = str(row[name]).strip().upper()
            if value in POSITIVE_STATUSES:
                return 1
            if value in NEGATIVE_STATUSES:
                return 0

    if "status" in row:
        value = str(row["status"]).strip().upper()
        if value in POSITIVE_STATUSES:
            return 1
        if value in NEGATIVE_STATUSES:
            return 0

    if "risk_score" in row:
        return int(parse_float(row, "risk_score") >= 50.0)
    return None


def row_to_features(row: dict[str, str]) -> np.ndarray:
    ttc = parse_float(row, "ttc_seconds", 0.0)
    time_to_impact = parse_float(row, "time_to_impact_zone", 0.0)
    frame_area = max(parse_float(row, "frame_area", 1.0), 1.0)

    values = {
        "ttc_inv": 0.0 if ttc <= 0.0 else 1.0 / max(ttc, 1e-3),
        "time_to_impact_inv": (
            0.0 if time_to_impact <= 0.0 else 1.0 / max(time_to_impact, 1e-3)
        ),
        "lane_distance": parse_float(row, "lane_distance"),
        "relative_distance": parse_float(row, "relative_distance"),
        "bbox_growth_rate": parse_float(row, "bbox_growth_rate"),
        "bbox_width_growth_rate": parse_float(row, "bbox_width_growth_rate"),
        "bbox_area_growth_rate_norm": parse_float(row, "bbox_area_growth_rate") / frame_area,
        "speed_px_s": parse_float(row, "speed_px_s"),
        "horizontal_speed_px_s_abs": abs(parse_float(row, "horizontal_speed_px_s")),
        "vertical_speed_px_s": parse_float(row, "vertical_speed_px_s"),
        "acceleration_px_s2": parse_float(row, "acceleration_px_s2"),
        "trajectory_intersection": parse_float(row, "trajectory_intersection"),
        "impact_zone_intersection": parse_float(row, "impact_zone_intersection"),
        "in_ego_corridor": parse_float(row, "in_ego_corridor"),
        "in_impact_zone": parse_float(row, "in_impact_zone"),
        "near_enough": parse_float(row, "near_enough"),
        "moving_toward_ego_center": parse_float(row, "moving_toward_ego_center"),
        "approaching_camera": parse_float(row, "approaching_camera"),
        "collision_candidate": parse_float(row, "collision_candidate"),
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=float)


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = parse_label(row)
                if label is None:
                    continue
                rows.append(row_to_features(row))
                labels.append(label)

    if not rows:
        raise ValueError("Tidak ada baris training valid. Butuh kolom label/status/risk_score.")

    y = np.array(labels, dtype=float)
    if len(set(labels)) < 2:
        raise ValueError("Training butuh minimal dua kelas: positif dan negatif.")
    return np.vstack(rows), y


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[float]]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_norm = (x - mean) / np.maximum(std, 1e-6)
    weights = np.zeros(x_norm.shape[1], dtype=float)
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        logits = np.clip(x_norm @ weights + bias, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y
        weights_gradient = (x_norm.T @ error) / len(y) + l2 * weights
        bias_gradient = float(error.mean())
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient

        loss = -np.mean(
            y * np.log(np.maximum(probabilities, 1e-8))
            + (1.0 - y) * np.log(np.maximum(1.0 - probabilities, 1e-8))
        )
        loss += 0.5 * l2 * float(np.dot(weights, weights))
        losses.append(float(loss))

    return weights, bias, mean, std, losses


def evaluate(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    x_norm = (x - mean) / np.maximum(std, 1e-6)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(x_norm @ weights + bias, -40.0, 40.0)))
    predictions = probabilities >= 0.5
    positives = y == 1.0
    negatives = y == 0.0
    tp = float(np.sum(predictions & positives))
    fp = float(np.sum(predictions & negatives))
    tn = float(np.sum(~predictions & negatives))
    fn = float(np.sum(~predictions & positives))
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model 2 risk ML dari CSV fitur hasil inference/labeling."
    )
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model2_risk_model.npz"),
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()

    x, y = load_dataset(args.features)
    weights, bias, mean, std, losses = train_logistic_regression(
        x=x,
        y=y,
        epochs=args.epochs,
        learning_rate=args.lr,
        l2=args.l2,
    )
    metrics = evaluate(x, y, weights, bias, mean, std)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=weights,
        bias=np.array(bias),
        mean=mean,
        std=std,
        feature_names=np.array(FEATURE_NAMES),
    )

    print(f"Output model : {args.output}")
    print(f"Rows         : {len(y)}")
    print(f"Final loss   : {losses[-1]:.4f}")
    print(f"Accuracy     : {metrics['accuracy']:.3f}")
    print(f"Precision    : {metrics['precision']:.3f}")
    print(f"Recall       : {metrics['recall']:.3f}")
    print(f"Positive rate: {metrics['positive_rate']:.3f}")


if __name__ == "__main__":
    main()
