from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
DATASET_CLASSES = ("positive", "negative")


@dataclass(frozen=True)
class VideoRecord:
    source_split: str
    dataset_class: str
    video_path: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration: float


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_records(input_dir: Path, source_splits: list[str]) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for source_split in source_splits:
        for dataset_class in DATASET_CLASSES:
            metadata_path = input_dir / source_split / dataset_class / "metadata.csv"
            if not metadata_path.exists():
                continue

            with metadata_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    file_name = row.get("file_name", "")
                    if not file_name:
                        continue
                    records.append(
                        VideoRecord(
                            source_split=source_split,
                            dataset_class=dataset_class,
                            video_path=metadata_path.parent / file_name,
                            metadata=row,
                        )
                    )
    return records


def read_video_info(video_path: Path) -> VideoInfo:
    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video tidak dapat dibuka: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0
        capture.release()
        return VideoInfo(fps=fps, frame_count=frame_count, duration=duration)

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "OpenCV tidak tersedia dan ffprobe tidak ditemukan. Install "
            "dependency dulu: pip install -r requirements.txt"
        )

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(numerator) / float(denominator or 1)
    duration = float(stream.get("duration") or 0.0)
    frame_count = int(stream.get("nb_frames") or round(duration * fps))
    return VideoInfo(fps=fps or 30.0, frame_count=frame_count, duration=duration)


def sample_frame_times(
    video_info: VideoInfo,
    record: VideoRecord,
    frames_per_video: int,
    positive_window_seconds: float,
) -> list[float]:
    fps = video_info.fps
    frame_count = video_info.frame_count
    if frame_count <= 0:
        return []

    if record.dataset_class == "positive":
        event_time = parse_float(record.metadata.get("time_of_event"))
        alert_time = parse_float(record.metadata.get("time_of_alert"))
        anchor_time = event_time if event_time is not None else alert_time
        if anchor_time is not None:
            start_time = max(0.0, anchor_time - positive_window_seconds)
            end_time = min(video_info.duration, anchor_time)
        else:
            start_time = 0.1 * video_info.duration
            end_time = 0.9 * video_info.duration
    else:
        start_time = 0.1 * video_info.duration
        end_time = 0.9 * video_info.duration

    if end_time <= start_time:
        return [min(max(start_time, 0.0), video_info.duration)]

    if frames_per_video == 1:
        times = [(start_time + end_time) / 2.0]
    else:
        step = (end_time - start_time) / max(frames_per_video - 1, 1)
        times = [start_time + step * index for index in range(frames_per_video)]

    return sorted({min(max(time_seconds, 0.0), video_info.duration) for time_seconds in times})


def extract_frame(video_path: Path, time_seconds: float, image_path: Path) -> bool:
    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return False
        capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return False
        return bool(cv2.imwrite(str(image_path), frame))

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "OpenCV tidak tersedia dan ffmpeg tidak ditemukan. Install "
            "dependency dulu: pip install -r requirements.txt"
        )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(image_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and image_path.exists()


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    names = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(class_names)
    )
    data_yaml = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    for relative_path in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        path = output_dir / relative_path
        if path.exists():
            shutil.rmtree(path)
    for relative_path in (
        Path("annotation_manifest.csv"),
        Path("data.yaml"),
        Path("labels/train.cache"),
        Path("labels/val.cache"),
    ):
        path = output_dir / relative_path
        if path.exists():
            path.unlink()


def extract_frames(
    records: list[VideoRecord],
    output_dir: Path,
    frames_per_video: int,
    positive_window_seconds: float,
    val_ratio: float,
    seed: int,
    class_names: list[str],
) -> None:
    existing_records = [record for record in records if record.video_path.exists()]
    if not existing_records:
        source_splits = sorted({record.source_split for record in records})
        raise FileNotFoundError(
            "Metadata ditemukan, tetapi tidak ada file video untuk split "
            f"{', '.join(source_splits)}. Unduh videonya lebih dulu, contoh: "
            "python download_nexar.py --source-splits train --max-videos 100"
        )

    rng = random.Random(seed)
    shuffled = existing_records[:]
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    val_videos = {record.video_path for record in shuffled[:val_count]}

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "annotation_manifest.csv"
    columns = [
        "image_path",
        "label_path",
        "output_split",
        "dataset_class",
        "source_split",
        "source_video",
        "frame_index",
        "time_seconds",
        "time_of_event",
        "time_of_alert",
        "light_conditions",
        "weather",
        "scene",
    ]

    extracted = 0
    skipped = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for record in shuffled:
            if not record.video_path.exists():
                skipped += 1
                continue

            try:
                video_info = read_video_info(record.video_path)
            except Exception:
                skipped += 1
                continue

            frame_times = sample_frame_times(
                video_info=video_info,
                record=record,
                frames_per_video=frames_per_video,
                positive_window_seconds=positive_window_seconds,
            )
            output_split = "val" if record.video_path in val_videos else "train"

            for time_seconds in frame_times:
                frame_index = min(
                    max(int(round(time_seconds * video_info.fps)), 0),
                    video_info.frame_count - 1,
                )
                stem = (
                    f"{record.source_split}_{record.dataset_class}_"
                    f"{record.video_path.stem}_f{frame_index:06d}"
                )
                image_path = output_dir / "images" / output_split / f"{stem}.jpg"
                label_path = output_dir / "labels" / output_split / f"{stem}.txt"

                ok = extract_frame(record.video_path, time_seconds, image_path)
                if not ok:
                    skipped += 1
                    continue

                label_path.touch()
                extracted += 1

                writer.writerow(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "output_split": output_split,
                        "dataset_class": record.dataset_class,
                        "source_split": record.source_split,
                        "source_video": record.video_path,
                        "frame_index": frame_index,
                        "time_seconds": round(time_seconds, 3),
                        "time_of_event": record.metadata.get("time_of_event", ""),
                        "time_of_alert": record.metadata.get("time_of_alert", ""),
                        "light_conditions": record.metadata.get(
                            "light_conditions",
                            "",
                        ),
                        "weather": record.metadata.get("weather", ""),
                        "scene": record.metadata.get("scene", ""),
                    }
                )

    write_data_yaml(output_dir, class_names)
    if extracted == 0:
        raise RuntimeError(
            "Tidak ada frame yang berhasil diekstrak. Pastikan file video "
            "dapat dibuka OpenCV dan tidak rusak."
        )

    print(f"Frame diekstrak : {extracted}")
    print(f"Video/file dilewati : {skipped}")
    print(f"Manifest anotasi : {manifest_path}")
    print(f"YOLO data.yaml : {output_dir / 'data.yaml'}")
    print("Catatan: label .txt masih kosong sampai frame diberi bounding box.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siapkan frame Nexar untuk anotasi dan training YOLO."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/nexar"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/yolo_nexar_frames"),
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["train"],
        help="Folder sumber Nexar, misalnya train atau test-public.",
    )
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--positive-window-seconds", type=float, default=4.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Hapus images/labels lama di output-dir sebelum ekstraksi.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Nama class YOLO untuk anotasi bounding box.",
    )
    args = parser.parse_args()

    records = load_records(args.input_dir, args.source_splits)
    if args.max_videos > 0:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: args.max_videos]
    if not records:
        raise FileNotFoundError(
            f"Tidak menemukan metadata Nexar di {args.input_dir}"
        )
    if args.frames_per_video <= 0:
        raise ValueError("--frames-per-video harus lebih dari 0")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio harus >= 0 dan < 1")

    if args.clean_output:
        clean_output_dir(args.output_dir)

    extract_frames(
        records=records,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        positive_window_seconds=args.positive_window_seconds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        class_names=args.classes,
    )


if __name__ == "__main__":
    main()
