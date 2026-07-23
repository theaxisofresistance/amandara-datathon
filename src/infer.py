from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from risk_engine import (
    TrackState,
    calculate_risk_from_features,
    extract_features,
    risk_category,
)

ROAD_USERS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
ROAD_USER_NAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def model_class_names(model: YOLO) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def road_user_class_ids(model: YOLO) -> list[int] | None:
    names = model_class_names(model)
    class_ids = [
        class_id
        for class_id, name in names.items()
        if name.lower() in ROAD_USER_NAMES
    ]
    if class_ids:
        return class_ids
    return list(ROAD_USERS)


def draw_tracked_alert(
    frame,
    bbox: tuple[float, float, float, float],
    points: list[tuple[float, float]],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    for index in range(1, len(points)):
        cv2.line(
            frame,
            tuple(map(int, points[index - 1])),
            tuple(map(int, points[index])),
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2,
    )
    cv2.putText(
        frame,
        label,
        (int(x1), max(25, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_risk_warning(
    frame,
    status: str,
) -> None:
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
        1.35,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Kendaraan menuju area ego dan semakin membesar",
        (36, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )



def process_video(
    input_path: Path,
    output_path: Path,
    events_path: Path,
    summary_path: Path,
    model_path: str,
    tracker: str,
    confidence: float,
    device: str | None,
) -> None:
    model = YOLO(model_path)
    class_names = model_class_names(model)
    class_filter = road_user_class_ids(model)
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

    states: dict[int, TrackState] = defaultdict(TrackState.create)
    last_event_frame: dict[int, int] = defaultdict(lambda: -100_000)
    events: list[dict[str, object]] = []
    frame_summaries: list[dict[str, object]] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=confidence,
            classes=class_filter,
            verbose=False,
            device=device,
        )[0]

        annotated = frame.copy()
        max_risk = 0.0
        max_risk_status = "SAFE"
        max_risk_object = ""
        max_risk_track_id = ""
        max_risk_ttc: float | None = None
        has_alert = False
        detected_object_count = 0
        collision_candidate_count = 0

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            if result.boxes.id is None:
                track_ids = [-(index + 1) for index in range(len(boxes))]
            else:
                track_ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

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

                points = []
                fallback_state = TrackState.create()
                if track_id >= 0:
                    state = states[track_id]
                    state.centers.append(center)
                    state.widths.append(bbox_width)
                    state.heights.append(bbox_height)
                    state.areas.append(bbox_area)
                    state.frames.append(frame_index)
                    points = list(state.centers)
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
                score, reasons = calculate_risk_from_features(
                    features=features,
                    bbox_height=bbox_height,
                    frame_height=height,
                )
                status, color = risk_category(score)
                detected_object_count += 1

                collision_candidate_count += int(features.collision_candidate)

                if score >= max_risk:
                    max_risk = score
                    max_risk_status = status
                    max_risk_object = class_names.get(
                        class_id,
                        ROAD_USERS.get(class_id, str(class_id)),
                    )
                    max_risk_track_id = "" if track_id < 0 else str(track_id)
                    max_risk_ttc = features.ttc

                if score >= 50:
                    has_alert = True

                draw_color = color
                object_name = class_names.get(
                    class_id,
                    ROAD_USERS.get(class_id, str(class_id)),
                )
                track_text = f" #{track_id}" if track_id >= 0 else ""
                label = f"{object_name}{track_text} | {status} {score:.0f}"
                draw_tracked_alert(
                    annotated,
                    (x1, y1, x2, y2),
                    points,
                    label,
                    draw_color,
                )

                cooldown = int(fps * 2)
                if (
                    track_id >= 0
                    and score >= 50
                    and frame_index - last_event_frame[track_id] >= cooldown
                ):
                    events.append(
                        {
                            "frame": frame_index,
                            "time_seconds": round(frame_index / fps, 3),
                            "track_id": track_id,
                            "object": object_name,
                            "detection_confidence": round(float(det_conf), 4),
                            "risk_score": round(score, 2),
                            "status": status,
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
                            "bbox_growth_rate": round(
                                features.bbox_growth_rate,
                                2,
                            ),
                            "bbox_width_growth_rate": round(
                                features.bbox_width_growth_rate,
                                2,
                            ),
                            "bbox_area_growth_rate": round(
                                features.bbox_area_growth_rate,
                                2,
                            ),
                            "trajectory_intersection": (
                                int(features.trajectory_intersection)
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
                            "approaching_camera": int(
                                features.approaching_camera
                            ),
                            "collision_candidate": int(
                                features.collision_candidate
                            ),
                            "reason": "; ".join(reasons),
                        }
                    )
                    last_event_frame[track_id] = frame_index

        frame_summaries.append(
            {
                "frame": frame_index,
                "time_seconds": round(frame_index / fps, 3),
                "object_count": collision_candidate_count,
                "detected_object_count": detected_object_count,
                "collision_candidate_count": collision_candidate_count,
                "max_risk_score": round(max_risk, 2),
                "max_risk_status": max_risk_status,
                "max_risk_object": max_risk_object,
                "max_risk_track_id": max_risk_track_id,
                "max_risk_ttc_seconds": (
                    "" if max_risk_ttc is None else round(max_risk_ttc, 3)
                ),
            }
        )

        cv2.rectangle(annotated, (14, 14), (390, 91), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            "AmanDara - Aman berkendara",
            (28, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Frame {frame_index}/{total_frames} | Max risk {max_risk:.0f}",
            (28, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            risk_category(max_risk)[1],
            2,
            cv2.LINE_AA,
        )

        if has_alert:
            draw_risk_warning(annotated, max_risk_status)


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
        csv_writer = csv.DictWriter(file, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(events)

    summary_columns = [
        "frame",
        "time_seconds",
        "object_count",
        "detected_object_count",
        "collision_candidate_count",
        "max_risk_score",
        "max_risk_status",
        "max_risk_object",
        "max_risk_track_id",
        "max_risk_ttc_seconds",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=summary_columns)
        csv_writer.writeheader()
        csv_writer.writerows(frame_summaries)

    print(f"Output video : {output_path}")
    print(f"Output CSV   : {events_path}")
    print(f"Summary CSV  : {summary_path}")
    print(f"Event risiko : {len(events)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/result.mp4"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("outputs/events.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/frame_summary.csv"),
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
    )
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, mps, atau indeks CUDA seperti 0.",
    )
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        events_path=args.events,
        summary_path=args.summary,
        model_path=args.model,
        tracker=args.tracker,
        confidence=args.conf,
        device=args.device,
    )


if __name__ == "__main__":
    main()
