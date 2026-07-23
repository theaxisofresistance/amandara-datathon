from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()

    videos = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.limit > 0:
        videos = videos[: args.limit]

    if not videos:
        raise FileNotFoundError(
            f"Tidak ada video ditemukan di {args.input_dir}"
        )

    for index, video in enumerate(videos, start=1):
        relative = video.relative_to(args.input_dir)
        output_video = args.output_dir / relative.with_suffix(".mp4")
        output_csv = args.output_dir / relative.with_suffix(".csv")
        output_summary = (
            args.output_dir
            / relative.with_name(f"{relative.stem}_frame_summary.csv")
        )
        output_video.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "src/infer.py",
            "--input",
            str(video),
            "--output",
            str(output_video),
            "--events",
            str(output_csv),
            "--summary",
            str(output_summary),
            "--model",
            args.model,
        ]
        if args.device:
            command.extend(["--device", args.device])

        print(f"[{index}/{len(videos)}] {video}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
