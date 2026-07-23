from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

REPO_ID = "nexar-ai/nexar_collision_prediction"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
METADATA_NAMES = (
    "metadata.csv",
    "metadata.jsonl",
    "metadata.json",
    "train.csv",
    "sample_submission.csv",
    "README.md",
    "LICENSE",
)


def is_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


def download_file(repo_path: str, target_dir: Path) -> Path:
    cached = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=repo_path,
        )
    )
    destination = target_dir / repo_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    return destination


def list_repository_files() -> list[str]:
    api = HfApi()
    return api.list_repo_files(REPO_ID, repo_type="dataset")


def download_subset(
    target_dir: Path,
    max_videos: int,
    start: int,
    source_splits: list[str],
) -> None:
    files = list_repository_files()
    selected_source_splits = tuple(f"{split.rstrip('/')}/" for split in source_splits)
    videos = sorted(
        path
        for path in files
        if is_video(path)
        and (
            not selected_source_splits
            or path.startswith(selected_source_splits)
        )
    )
    metadata = sorted(
        path
        for path in files
        if (
            not selected_source_splits
            or path.startswith(selected_source_splits)
        )
        and (
            Path(path).name in METADATA_NAMES
            or "metadata" in Path(path).name.lower()
        )
    )

    if not videos:
        raise RuntimeError("Tidak menemukan file video pada repository.")

    selected = videos[start:]
    if max_videos > 0:
        selected = selected[:max_videos]

    print(f"Total video repository : {len(videos)}")
    print(f"Video yang diunduh     : {len(selected)}")
    print(f"Mulai dari indeks      : {start}")
    print(f"Source split           : {', '.join(source_splits)}")

    failures: list[dict[str, str]] = []

    for path in tqdm(metadata, desc="Metadata"):
        try:
            download_file(path, target_dir)
        except Exception as error:
            failures.append({"file": path, "error": str(error)})

    for path in tqdm(selected, desc="Video"):
        try:
            download_file(path, target_dir)
        except Exception as error:
            failures.append({"file": path, "error": str(error)})

    manifest = {
        "repo_id": REPO_ID,
        "total_repository_videos": len(videos),
        "source_splits": source_splits,
        "start": start,
        "requested_videos": max_videos,
        "selected_files": selected,
        "failures": failures,
    }
    manifest_path = target_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nDataset tersimpan : {target_dir}")
    print(f"Manifest          : {manifest_path}")
    print(f"Gagal             : {len(failures)}")


def download_full(target_dir: Path) -> None:
    print("Mengunduh snapshot penuh Nexar. Ukurannya sekitar 31 GB.")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=target_dir,
    )
    print(f"Dataset lengkap tersimpan: {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downloader Nexar Collision Prediction dari Hugging Face."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/nexar"),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=20,
        help="Jumlah video subset. Gunakan 0 untuk seluruh daftar via mode subset.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Indeks awal video untuk melanjutkan batch berikutnya.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["train"],
        help="Folder sumber yang diunduh, misalnya train atau test-public.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Unduh seluruh snapshot dataset.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.full:
        download_full(args.output)
    else:
        download_subset(
            target_dir=args.output,
            max_videos=args.max_videos,
            start=args.start,
            source_splits=args.source_splits,
        )


if __name__ == "__main__":
    main()
