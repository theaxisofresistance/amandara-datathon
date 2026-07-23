from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class TrackState:
    centers: Deque[tuple[float, float]]
    widths: Deque[float]
    heights: Deque[float]
    areas: Deque[float]
    frames: Deque[int]

    @classmethod
    def create(cls, history: int = 24) -> "TrackState":
        return cls(
            centers=deque(maxlen=history),
            widths=deque(maxlen=history),
            heights=deque(maxlen=history),
            areas=deque(maxlen=history),
            frames=deque(maxlen=history),
        )


@dataclass(frozen=True)
class RiskFeatures:
    ttc: float | None
    time_to_ego_zone: float | None
    time_to_impact_zone: float | None
    lane_distance: float
    bbox_growth_rate: float
    bbox_width_growth_rate: float
    bbox_area_growth_rate: float
    speed_px_s: float
    horizontal_speed_px_s: float
    vertical_speed_px_s: float
    acceleration_px_s2: float
    trajectory_intersection: bool
    in_ego_corridor: bool
    impact_zone_intersection: bool
    in_impact_zone: bool
    edge_intrusion: bool
    near_enough: bool
    moving_toward_ego_center: bool
    approaching_camera: bool
    collision_candidate: bool
    relative_distance: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def euclidean(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return math.sqrt(dx * dx + dy * dy)


def estimate_ttc(
    heights: Deque[float],
    frames: Deque[int],
    fps: float,
) -> float | None:
    """
    TTC relatif dari pertumbuhan tinggi bounding box.

    Nilai ini bukan jarak fisik terkalibrasi. Tujuannya adalah memberi
    indikator apakah objek membesar cepat di depan kendaraan ego.
    """
    if len(heights) < 5:
        return None

    first = max(float(heights[0]), 1.0)
    last = max(float(heights[-1]), 1.0)
    elapsed = max((frames[-1] - frames[0]) / max(fps, 1e-6), 1e-3)
    growth_per_second = (last - first) / elapsed

    if growth_per_second <= 1e-6:
        return None

    ttc = last / growth_per_second
    if not math.isfinite(ttc) or ttc <= 0:
        return None
    return float(ttc)


def estimate_motion(
    centers: Deque[tuple[float, float]],
    frames: Deque[int],
    fps: float,
) -> tuple[float, float, float, float]:
    if len(centers) < 2 or len(frames) < 2:
        return 0.0, 0.0, 0.0, 0.0

    speeds: list[float] = []
    for index in range(1, len(centers)):
        elapsed = max((frames[index] - frames[index - 1]) / max(fps, 1e-6), 1e-3)
        speeds.append(euclidean(centers[index], centers[index - 1]) / elapsed)

    speed = speeds[-1]
    elapsed = max((frames[-1] - frames[-2]) / max(fps, 1e-6), 1e-3)
    horizontal_speed = (centers[-1][0] - centers[-2][0]) / elapsed
    vertical_speed = (centers[-1][1] - centers[-2][1]) / elapsed

    if len(speeds) < 2:
        return speed, horizontal_speed, vertical_speed, 0.0

    acceleration = (speeds[-1] - speeds[-2]) / elapsed
    return speed, horizontal_speed, vertical_speed, acceleration


def trajectory_to_ego_zone(
    centers: Deque[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    fps: float,
    horizon_frames: int = 30,
) -> tuple[bool, float | None, bool]:
    if len(centers) < 4:
        return False, None, False

    start = centers[-4]
    end = centers[-1]
    vx = (end[0] - start[0]) / 3.0
    vy = (end[1] - start[1]) / 3.0
    if abs(vx) + abs(vy) < 1e-3:
        return False, None, False

    ego_center_x = frame_width * 0.5
    previous_distance = abs(start[0] - ego_center_x)
    current_distance = abs(end[0] - ego_center_x)
    moving_toward_center = current_distance < previous_distance

    for step in range(1, horizon_frames + 1):
        px = end[0] + vx * step
        py = end[1] + vy * step
        if point_in_ego_corridor((px, py), frame_width, frame_height):
            return True, step / max(fps, 1e-6), moving_toward_center
    return False, None, moving_toward_center


def trajectory_to_impact_zone(
    centers: Deque[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    fps: float,
    horizon_frames: int = 30,
) -> tuple[bool, float | None]:
    if len(centers) < 4:
        return False, None

    start = centers[-4]
    end = centers[-1]
    vx = (end[0] - start[0]) / 3.0
    vy = (end[1] - start[1]) / 3.0
    if abs(vx) + abs(vy) < 1e-3:
        return False, None

    for step in range(1, horizon_frames + 1):
        px = end[0] + vx * step
        py = end[1] + vy * step
        if point_in_impact_zone((px, py), frame_width, frame_height):
            return True, step / max(fps, 1e-6)
    return False, None


def point_in_ego_corridor(
    center: tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    """
    Area ego adalah C: bagian bawah-tengah frame, dekat kendaraan/kamera.

    Area A dan B bukan ego area. Objek di A/B baru dianggap RISK kalau
    trajectory-nya diproyeksikan masuk ke area C sambil bbox membesar.
    """
    x, y = center
    x_norm = x / max(frame_width, 1)
    y_norm = y / max(frame_height, 1)
    return 0.18 <= x_norm <= 0.82 and 0.76 <= y_norm <= 1.02


def point_in_impact_zone(
    center: tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    return point_in_ego_corridor(center, frame_width, frame_height)


def bbox_edge_intrusion(
    center: tuple[float, float],
    bbox_width: float,
    bbox_height: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    x, y = center
    x1 = x - bbox_width / 2.0
    x2 = x + bbox_width / 2.0
    y2 = y + bbox_height / 2.0

    touches_side = x1 <= frame_width * 0.04 or x2 >= frame_width * 0.96
    extends_into_frame = x2 >= frame_width * 0.14 and x1 <= frame_width * 0.86
    close_to_ego = y2 >= frame_height * 0.76
    large_enough = (
        bbox_height / max(frame_height, 1) >= 0.16
        and bbox_width / max(frame_width, 1) >= 0.16
    )
    return touches_side and extends_into_frame and close_to_ego and large_enough


def horizontal_motion_toward_center(
    center: tuple[float, float],
    horizontal_speed: float,
    frame_width: int,
    min_speed: float = 90.0,
) -> bool:
    center_x = frame_width * 0.5
    if center[0] < center_x:
        return horizontal_speed > min_speed
    if center[0] > center_x:
        return horizontal_speed < -min_speed
    return False


def object_is_near_enough(
    center: tuple[float, float],
    bbox_height: float,
    frame_height: int,
) -> bool:
    y_norm = center[1] / max(frame_height, 1)
    height_ratio = bbox_height / max(frame_height, 1)
    return height_ratio >= 0.09 or y_norm >= 0.62


def extract_features(
    state: TrackState,
    center: tuple[float, float],
    bbox_width: float,
    bbox_height: float,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> RiskFeatures:
    ttc = estimate_ttc(state.heights, state.frames, fps)
    speed, horizontal_speed, vertical_speed, acceleration = estimate_motion(
        state.centers,
        state.frames,
        fps,
    )

    conflict_point = (frame_width * 0.5, frame_height * 0.90)
    dx = (center[0] - conflict_point[0]) / max(frame_width, 1)
    dy = (center[1] - conflict_point[1]) / max(frame_height, 1)
    lane_distance = math.sqrt(dx * dx + dy * dy)

    growth_rate = 0.0
    if len(state.heights) >= 2:
        first = max(float(state.heights[0]), 1.0)
        last = max(float(state.heights[-1]), 1.0)
        elapsed = max((state.frames[-1] - state.frames[0]) / max(fps, 1e-6), 1e-3)
        growth_rate = (last - first) / elapsed

    width_growth_rate = 0.0
    if len(state.widths) >= 2:
        first = max(float(state.widths[0]), 1.0)
        last = max(float(state.widths[-1]), 1.0)
        elapsed = max((state.frames[-1] - state.frames[0]) / max(fps, 1e-6), 1e-3)
        width_growth_rate = (last - first) / elapsed

    area_growth_rate = 0.0
    if len(state.areas) >= 2:
        first = max(float(state.areas[0]), 1.0)
        last = max(float(state.areas[-1]), 1.0)
        elapsed = max((state.frames[-1] - state.frames[0]) / max(fps, 1e-6), 1e-3)
        area_growth_rate = (last - first) / elapsed

    in_corridor = point_in_ego_corridor(center, frame_width, frame_height)
    in_impact_zone = point_in_impact_zone(center, frame_width, frame_height)
    edge_intrusion = bbox_edge_intrusion(
        center,
        bbox_width,
        bbox_height,
        frame_width,
        frame_height,
    )
    trajectory_intersection, time_to_zone, moving_toward_center = trajectory_to_ego_zone(
        state.centers,
        frame_width,
        frame_height,
        fps,
    )
    impact_intersection, time_to_impact = trajectory_to_impact_zone(
        state.centers,
        frame_width,
        frame_height,
        fps,
    )
    frame_area = max(frame_width * frame_height, 1)
    normalized_area_growth = area_growth_rate / frame_area
    approaching_camera = (
        growth_rate > 45.0
        or width_growth_rate > 90.0
        or normalized_area_growth > 0.018
        or (ttc is not None and ttc < 3.0)
    )
    near_enough = object_is_near_enough(center, bbox_height, frame_height)
    x_norm = center[0] / max(frame_width, 1)
    vertical_motion_toward_c = 0.18 <= x_norm <= 0.82 and vertical_speed > 90.0
    side_motion_toward_center = (
        moving_toward_center
        or horizontal_motion_toward_center(center, horizontal_speed, frame_width)
        or vertical_motion_toward_c
    )
    crossing_motion = (
        impact_intersection
        and side_motion_toward_center
        and near_enough
        and speed > 160.0
        and approaching_camera
        and time_to_impact is not None
        and time_to_impact <= 3.0
    )
    frontal_collision_course = (
        in_impact_zone
        and near_enough
        and approaching_camera
    )
    return RiskFeatures(
        ttc=ttc,
        time_to_ego_zone=time_to_zone,
        time_to_impact_zone=time_to_impact,
        lane_distance=lane_distance,
        bbox_growth_rate=growth_rate,
        bbox_width_growth_rate=width_growth_rate,
        bbox_area_growth_rate=area_growth_rate,
        speed_px_s=speed,
        horizontal_speed_px_s=horizontal_speed,
        vertical_speed_px_s=vertical_speed,
        acceleration_px_s2=acceleration,
        trajectory_intersection=trajectory_intersection,
        in_ego_corridor=in_corridor,
        impact_zone_intersection=impact_intersection,
        in_impact_zone=in_impact_zone,
        edge_intrusion=edge_intrusion,
        near_enough=near_enough,
        moving_toward_ego_center=side_motion_toward_center,
        approaching_camera=approaching_camera,
        collision_candidate=(
            frontal_collision_course
            or crossing_motion
        ),
        relative_distance=1.0 - clamp(bbox_height / max(frame_height, 1), 0.0, 1.0),
    )


def calculate_risk_from_features(
    features: RiskFeatures,
    bbox_height: float,
    frame_height: int,
) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if not features.near_enough:
        return 0.0, ["SAFE: objek masih terlalu jauh"]

    if not features.approaching_camera:
        return 0.0, ["SAFE: bbox tidak membesar / objek menjauh"]

    if (
        not features.in_impact_zone
        and not features.impact_zone_intersection
    ):
        return 0.0, ["SAFE: membesar tetapi tidak menuju zona benturan ego"]

    if (
        not features.in_ego_corridor
        and not features.trajectory_intersection
    ):
        return 0.0, ["SAFE: di luar koridor ego"]

    if not features.collision_candidate:
        return 0.0, ["SAFE: tidak menuju ego sambil membesar"]

    reasons.append("RISK: menuju zona ego dan bbox membesar")
    if features.trajectory_intersection:
        reasons.append("trajectory masuk zona ego")
    if features.in_ego_corridor:
        reasons.append("di koridor ego")
    if features.in_impact_zone:
        reasons.append("di zona benturan ego")
    if features.impact_zone_intersection:
        reasons.append("trajectory menuju zona benturan ego")
    if features.edge_intrusion:
        reasons.append("kendaraan masuk dari tepi dekat ego")
    if features.moving_toward_ego_center:
        reasons.append("bergerak menuju center ego")
    if features.approaching_camera:
        reasons.append("ukuran bbox membesar / mendekat")
    return 100.0, reasons


def calculate_risk(
    center: tuple[float, float],
    bbox_height: float,
    frame_width: int,
    frame_height: int,
    ttc: float | None,
) -> tuple[float, list[str]]:
    conflict_point = (frame_width * 0.5, frame_height * 0.90)
    dx = (center[0] - conflict_point[0]) / max(frame_width, 1)
    dy = (center[1] - conflict_point[1]) / max(frame_height, 1)
    features = RiskFeatures(
        ttc=ttc,
        time_to_ego_zone=None,
        time_to_impact_zone=None,
        lane_distance=math.sqrt(dx * dx + dy * dy),
        bbox_growth_rate=0.0,
        bbox_width_growth_rate=0.0,
        bbox_area_growth_rate=0.0,
        speed_px_s=0.0,
        horizontal_speed_px_s=0.0,
        vertical_speed_px_s=0.0,
        acceleration_px_s2=0.0,
        trajectory_intersection=False,
        in_ego_corridor=False,
        impact_zone_intersection=False,
        in_impact_zone=False,
        edge_intrusion=False,
        near_enough=bbox_height / max(frame_height, 1) >= 0.09,
        moving_toward_ego_center=False,
        approaching_camera=False,
        collision_candidate=False,
        relative_distance=1.0 - clamp(bbox_height / max(frame_height, 1), 0.0, 1.0),
    )
    return calculate_risk_from_features(features, bbox_height, frame_height)


def risk_category(score: float) -> tuple[str, tuple[int, int, int]]:
    if score >= 50:
        return "RISK", (0, 0, 255)
    return "SAFE", (0, 200, 0)
