from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
POSITIVE_NAMES = {"positive", "collision", "near_collision", "near-collision"}
NEGATIVE_NAMES = {"negative", "normal", "normal_driving", "normal-driving"}


@dataclass(frozen=True)
class VideoPrediction:
    video_path: Path
    prediction_path: Path | None
    label: int
    score: float
    detected_events: int
    predicted: int
    status: str


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def label_from_path(video_path: Path) -> int | None:
    parts = {part.lower() for part in video_path.parts}
    if parts & POSITIVE_NAMES:
        return 1
    if parts & NEGATIVE_NAMES:
        return 0
    return None


def discover_videos(dataset_dir: Path) -> list[Path]:
    videos = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"Tidak ada video ditemukan di {dataset_dir}")
    return videos


def candidate_prediction_paths(
    video_path: Path,
    dataset_dir: Path,
    predictions_dir: Path,
) -> list[Path]:
    relative = video_path.relative_to(dataset_dir).with_suffix("")
    return [
        predictions_dir / relative.with_name(f"{relative.name}_frame_summary.csv"),
        predictions_dir / relative.with_suffix(".csv"),
        predictions_dir / f"{video_path.stem}_frame_summary.csv",
        predictions_dir / f"{video_path.stem}.csv",
    ]


def read_summary_score(path: Path) -> tuple[float, int]:
    max_score = 0.0
    high_risk_frames = 0
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            score = parse_float(row.get("max_risk_score"))
            max_score = max(max_score, score)
            high_risk_frames += int(score >= 50.0)
    return max_score, high_risk_frames


def read_event_score(path: Path) -> tuple[float, int]:
    max_score = 0.0
    events = 0
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            score = parse_float(row.get("risk_score"))
            max_score = max(max_score, score)
            events += 1
    return max_score, events


def read_prediction_score(path: Path) -> tuple[float, int]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

    if "max_risk_score" in fieldnames:
        return read_summary_score(path)
    if "risk_score" in fieldnames:
        return read_event_score(path)
    raise ValueError(f"CSV tidak punya kolom skor yang dikenal: {path}")


def evaluate_video(
    video_path: Path,
    dataset_dir: Path,
    predictions_dir: Path,
    threshold: float,
    missing_score: float,
) -> VideoPrediction | None:
    label = label_from_path(video_path)
    if label is None:
        return None

    prediction_path = None
    score = missing_score
    detected_events = 0
    for candidate in candidate_prediction_paths(video_path, dataset_dir, predictions_dir):
        if candidate.exists():
            prediction_path = candidate
            score, detected_events = read_prediction_score(candidate)
            break

    predicted = int(score >= threshold)
    if label == 1 and predicted == 1:
        status = "TP"
    elif label == 0 and predicted == 1:
        status = "FP"
    elif label == 0 and predicted == 0:
        status = "TN"
    else:
        status = "FN"

    return VideoPrediction(
        video_path=video_path,
        prediction_path=prediction_path,
        label=label,
        score=score,
        detected_events=detected_events,
        predicted=predicted,
        status=status,
    )


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def summarize(predictions: list[VideoPrediction]) -> dict[str, float | int]:
    tp = sum(item.status == "TP" for item in predictions)
    fp = sum(item.status == "FP" for item in predictions)
    tn = sum(item.status == "TN" for item in predictions)
    fn = sum(item.status == "FN" for item in predictions)
    return metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)


def threshold_sweep(
    predictions: list[VideoPrediction],
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for item in predictions:
            predicted = int(item.score >= threshold)
            if item.label == 1 and predicted == 1:
                tp += 1
            elif item.label == 0 and predicted == 1:
                fp += 1
            elif item.label == 0 and predicted == 0:
                tn += 1
            else:
                fn += 1
        metrics = metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)
        metrics["threshold"] = threshold
        rows.append(metrics)
    return rows


def write_predictions(path: Path, predictions: list[VideoPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "video_path",
        "prediction_path",
        "label",
        "score",
        "detected_events",
        "predicted",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in predictions:
            writer.writerow(
                {
                    "video_path": item.video_path,
                    "prediction_path": "" if item.prediction_path is None else item.prediction_path,
                    "label": item.label,
                    "score": round(item.score, 4),
                    "detected_events": item.detected_events,
                    "predicted": item.predicted,
                    "status": item.status,
                }
            )


def write_sweep(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "threshold",
        "total",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluasi video-level untuk Model 1 dari output batch CSV."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/nexar"))
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/batch"),
        help="Folder output dari src/batch_process.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Video diprediksi positif jika max_risk_score >= threshold.",
    )
    parser.add_argument(
        "--missing-score",
        type=float,
        default=0.0,
        help="Score untuk video yang belum punya CSV prediksi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model1_evaluation.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/model1_evaluation_summary.json"),
    )
    parser.add_argument(
        "--sweep-output",
        type=Path,
        default=Path("outputs/model1_threshold_sweep.csv"),
    )
    parser.add_argument(
        "--sweep",
        default="0,25,50,75",
        help="Daftar threshold untuk sweep, dipisah koma.",
    )
    args = parser.parse_args()

    videos = discover_videos(args.dataset_dir)
    predictions = [
        prediction
        for video in videos
        if (
            prediction := evaluate_video(
                video_path=video,
                dataset_dir=args.dataset_dir,
                predictions_dir=args.predictions_dir,
                threshold=args.threshold,
                missing_score=args.missing_score,
            )
        )
        is not None
    ]
    if not predictions:
        raise RuntimeError(
            "Tidak ada video yang bisa diberi label. Pastikan path mengandung folder "
            "positive/negative atau collision/normal."
        )

    summary = summarize(predictions)
    missing_count = sum(item.prediction_path is None for item in predictions)
    summary["threshold"] = args.threshold
    summary["missing_predictions"] = missing_count
    summary["dataset_dir"] = str(args.dataset_dir)
    summary["predictions_dir"] = str(args.predictions_dir)

    write_predictions(args.output, predictions)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    thresholds = [
        parse_float(value.strip())
        for value in args.sweep.split(",")
        if value.strip()
    ]
    write_sweep(args.sweep_output, threshold_sweep(predictions, thresholds))

    print(f"Videos evaluated   : {summary['total']}")
    print(f"Missing predictions: {missing_count}")
    print(f"Threshold          : {args.threshold:.2f}")
    print(
        "Confusion          : "
        f"TP={summary['tp']} FP={summary['fp']} TN={summary['tn']} FN={summary['fn']}"
    )
    print(f"Accuracy           : {summary['accuracy']:.3f}")
    print(f"Precision          : {summary['precision']:.3f}")
    print(f"Recall             : {summary['recall']:.3f}")
    print(f"Specificity        : {summary['specificity']:.3f}")
    print(f"F1                 : {summary['f1']:.3f}")
    print(f"Per-video output   : {args.output}")
    print(f"Summary JSON       : {args.summary}")
    print(f"Threshold sweep    : {args.sweep_output}")


if __name__ == "__main__":
    main()
