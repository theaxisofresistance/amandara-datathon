from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from infer import ROAD_USERS, draw_tracked_alert, model_class_names, road_user_class_ids
from model2_risk_engine import Model2RiskPredictor
from risk_engine import TrackState, extract_features, risk_category


def draw_model2_alert(frame, status: str, source: str) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width, min(132, height)),
        (0, 0, 210),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        status,
        (32, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Model 2 | Fine-tuned YOLO + ML risk ({source})",
        (36, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_video_model2(
    input_path: Path,
    output_path: Path,
    events_path: Path,
    summary_path: Path,
    detector_path: str,
    risk_model_path: Path | None,
    tracker: str,
    confidence: float,
    device: str | None,
    no_rule_fallback: bool,
    allow_pretrained_detector_fallback: bool,
) -> None:
    detector_file = Path(detector_path)
    if not detector_file.exists():
        fallback_detector = Path("yolo11s.pt")
        if allow_pretrained_detector_fallback and fallback_detector.exists():
            print(
                "Detector fine-tuned belum ada, memakai yolo11s.pt untuk testing. "
                "Ini belum Model 2 penuh."
            )
            detector_path = str(fallback_detector)
        else:
            raise FileNotFoundError(
                "\n".join(
                    [
                        f"Detector Model 2 tidak ditemukan: {detector_path}",
                        "Train YOLO fine-tuned dulu sampai menghasilkan best.pt, atau",
                        "jalankan testing sementara dengan:",
                        "  --detector yolo11s.pt",
                        "atau:",
                        "  --allow-pretrained-detector-fallback",
                    ]
                )
            )

    detector = YOLO(detector_path)
    class_names = model_class_names(detector)
    class_filter = road_user_class_ids(detector)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video tidak dapat dibuka: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Tidak dapat membuat output: {output_path}")

    risk_predictor = Model2RiskPredictor(
        model_path=risk_model_path,
        frame_width=width,
        frame_height=height,
        fallback_to_rules=not no_rule_fallback,
    )

    states: dict[int, TrackState] = defaultdict(TrackState.create)
    last_event_frame: dict[int, int] = defaultdict(lambda: -100_000)
    events: list[dict[str, object]] = []
    frame_summaries: list[dict[str, object]] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = detector.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=confidence,
            classes=class_filter,
            verbose=False,
            device=device,
        )[0]

        annotated = frame.copy()
        max_score = 0.0
        max_status = "SAFE"
        max_object = ""
        max_track_id = ""
        max_ttc: float | None = None
        max_source = "ml" if risk_predictor.has_ml_model else "rule_fallback"
        has_alert = False
        detected_object_count = 0
        high_risk_count = 0

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            if result.boxes.id is None:
                track_ids = [-(index + 1) for index in range(len(boxes))]
            else:
                track_ids = result.boxes.id.int().cpu().tolist()

            for box, class_id, track_id, det_conf in zip(
                boxes,
                class_ids,
                track_ids,
                confidences,
            ):
                x1, y1, x2, y2 = map(float, box)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                bbox_width = max(x2 - x1, 1.0)
                bbox_height = max(y2 - y1, 1.0)
                bbox_area = bbox_width * bbox_height

                fallback_state = TrackState.create()
                if track_id >= 0:
                    state = states[track_id]
                else:
                    state = fallback_state
                state.centers.append(center)
                state.widths.append(bbox_width)
                state.heights.append(bbox_height)
                state.areas.append(bbox_area)
                state.frames.append(frame_index)

                features = extract_features(
                    state=state,
                    center=center,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    frame_width=width,
                    frame_height=height,
                    fps=fps,
                )
                prediction = risk_predictor.predict(
                    features=features,
                    bbox_height=bbox_height,
                    frame_height=height,
                )

                detected_object_count += 1
                high_risk_count += int(prediction.score >= 50.0)
                if prediction.score >= max_score:
                    max_score = prediction.score
                    max_status = prediction.status
                    max_object = class_names.get(
                        class_id,
                        ROAD_USERS.get(class_id, str(class_id)),
                    )
                    max_track_id = "" if track_id < 0 else str(track_id)
                    max_ttc = features.ttc
                    max_source = prediction.source
                if prediction.score >= 50.0:
                    has_alert = True

                object_name = class_names.get(
                    class_id,
                    ROAD_USERS.get(class_id, str(class_id)),
                )
                track_text = f" #{track_id}" if track_id >= 0 else ""
                label = f"{object_name}{track_text} | {prediction.status} {prediction.score:.0f}"
                draw_tracked_alert(
                    annotated,
                    (x1, y1, x2, y2),
                    list(state.centers),
                    label,
                    prediction.color,
                )

                cooldown = int(fps * 2)
                if (
                    track_id >= 0
                    and prediction.score >= 50.0
                    and frame_index - last_event_frame[track_id] >= cooldown
                ):
                    events.append(
                        {
                            "frame": frame_index,
                            "time_seconds": round(frame_index / fps, 3),
                            "track_id": track_id,
                            "object": object_name,
                            "detection_confidence": round(float(det_conf), 4),
                            "risk_score": round(prediction.score, 2),
                            "status": prediction.status,
                            "risk_source": prediction.source,
                            "ttc_seconds": (
                                "" if features.ttc is None else round(features.ttc, 3)
                            ),
                            "lane_distance": round(features.lane_distance, 4),
                            "relative_distance": round(features.relative_distance, 4),
                            "speed_px_s": round(features.speed_px_s, 2),
                            "horizontal_speed_px_s": round(
                                features.horizontal_speed_px_s,
                                2,
                            ),
                            "vertical_speed_px_s": round(
                                features.vertical_speed_px_s,
                                2,
                            ),
                            "acceleration_px_s2": round(
                                features.acceleration_px_s2,
                                2,
                            ),
                            "bbox_growth_rate": round(features.bbox_growth_rate, 2),
                            "bbox_width_growth_rate": round(
                                features.bbox_width_growth_rate,
                                2,
                            ),
                            "bbox_area_growth_rate": round(
                                features.bbox_area_growth_rate,
                                2,
                            ),
                            "trajectory_intersection": int(
                                features.trajectory_intersection
                            ),
                            "time_to_ego_zone": (
                                ""
                                if features.time_to_ego_zone is None
                                else round(features.time_to_ego_zone, 3)
                            ),
                            "time_to_impact_zone": (
                                ""
                                if features.time_to_impact_zone is None
                                else round(features.time_to_impact_zone, 3)
                            ),
                            "in_ego_corridor": int(features.in_ego_corridor),
                            "in_impact_zone": int(features.in_impact_zone),
                            "impact_zone_intersection": int(
                                features.impact_zone_intersection
                            ),
                            "edge_intrusion": int(features.edge_intrusion),
                            "near_enough": int(features.near_enough),
                            "moving_toward_ego_center": int(
                                features.moving_toward_ego_center
                            ),
                            "approaching_camera": int(features.approaching_camera),
                            "collision_candidate": int(features.collision_candidate),
                            "reason": "; ".join(prediction.reasons),
                        }
                    )
                    last_event_frame[track_id] = frame_index

        frame_summaries.append(
            {
                "frame": frame_index,
                "time_seconds": round(frame_index / fps, 3),
                "object_count": high_risk_count,
                "detected_object_count": detected_object_count,
                "high_risk_count": high_risk_count,
                "max_risk_score": round(max_score, 2),
                "max_risk_status": max_status,
                "max_risk_object": max_object,
                "max_risk_track_id": max_track_id,
                "max_risk_ttc_seconds": "" if max_ttc is None else round(max_ttc, 3),
                "risk_source": max_source,
            }
        )

        cv2.rectangle(annotated, (14, 14), (430, 91), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            "AmanDara - Model 2",
            (28, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Frame {frame_index}/{total_frames} | Max risk {max_score:.0f}",
            (28, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            risk_category(max_score)[1],
            2,
            cv2.LINE_AA,
        )
        if has_alert:
            draw_model2_alert(annotated, max_status, max_source)

        writer.write(annotated)
        frame_index += 1

    capture.release()
    writer.release()

    columns = [
        "frame",
        "time_seconds",
        "track_id",
        "object",
        "detection_confidence",
        "risk_score",
        "status",
        "risk_source",
        "ttc_seconds",
        "lane_distance",
        "relative_distance",
        "speed_px_s",
        "horizontal_speed_px_s",
        "vertical_speed_px_s",
        "acceleration_px_s2",
        "bbox_growth_rate",
        "bbox_width_growth_rate",
        "bbox_area_growth_rate",
        "trajectory_intersection",
        "time_to_ego_zone",
        "time_to_impact_zone",
        "in_ego_corridor",
        "in_impact_zone",
        "impact_zone_intersection",
        "edge_intrusion",
        "near_enough",
        "moving_toward_ego_center",
        "approaching_camera",
        "collision_candidate",
        "reason",
    ]
    with events_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=columns)
        writer_csv.writeheader()
        writer_csv.writerows(events)

    summary_columns = [
        "frame",
        "time_seconds",
        "object_count",
        "detected_object_count",
        "high_risk_count",
        "max_risk_score",
        "max_risk_status",
        "max_risk_object",
        "max_risk_track_id",
        "max_risk_ttc_seconds",
        "risk_source",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=summary_columns)
        writer_csv.writeheader()
        writer_csv.writerows(frame_summaries)

    print(f"Output video : {output_path}")
    print(f"Output CSV   : {events_path}")
    print(f"Summary CSV  : {summary_path}")
    print(f"Risk source  : {'ml' if risk_predictor.has_ml_model else 'rule_fallback'}")
    print(f"Event risiko : {len(events)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model 2: fine-tuned YOLO + ML risk predictor."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model2_result.mp4"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("outputs/model2_events.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/model2_frame_summary.csv"),
    )
    parser.add_argument(
        "--detector",
        default="runs/detect/model2_nexar/weights/best.pt",
        help="Path weights YOLO fine-tuned Nexar, biasanya best.pt.",
    )
    parser.add_argument(
        "--risk-model",
        type=Path,
        default=Path("models/model2_risk_model.npz"),
        help="Path model risk ML hasil src/train_model2_risk.py.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, mps, atau indeks CUDA seperti 0.",
    )
    parser.add_argument(
        "--no-rule-fallback",
        action="store_true",
        help="Error jika --risk-model belum ada, bukan fallback ke rule Model 1.",
    )
    parser.add_argument(
        "--allow-pretrained-detector-fallback",
        action="store_true",
        help="Pakai yolo11s.pt untuk testing jika best.pt Model 2 belum ada.",
    )
    args = parser.parse_args()

    process_video_model2(
        input_path=args.input,
        output_path=args.output,
        events_path=args.events,
        summary_path=args.summary,
        detector_path=args.detector,
        risk_model_path=args.risk_model,
        tracker=args.tracker,
        confidence=args.conf,
        device=args.device,
        no_rule_fallback=args.no_rule_fallback,
        allow_pretrained_detector_fallback=args.allow_pretrained_detector_fallback,
    )


if __name__ == "__main__":
    main()
