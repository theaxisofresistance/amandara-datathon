from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COCO_TO_LOCAL = {
    0: 0,  # person
    1: 1,  # bicycle
    2: 2,  # car
    3: 3,  # motorcycle
    5: 4,  # bus
    7: 5,  # truck
}


def image_paths(dataset_dir: Path, split: str) -> list[Path]:
    image_dir = dataset_dir / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {image_dir}")
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(dataset_dir: Path, split: str, image_path: Path) -> Path:
    image_dir = dataset_dir / "images" / split
    relative = image_path.relative_to(image_dir).with_suffix(".txt")
    return dataset_dir / "labels" / split / relative


def write_preview(result, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(str(preview_path), annotated)


def autolabel_split(
    model: YOLO,
    dataset_dir: Path,
    split: str,
    confidence: float,
    iou: float,
    image_size: int,
    batch_size: int,
    device: str | None,
    overwrite: bool,
    preview_dir: Path | None,
    max_images: int,
) -> tuple[int, int, int]:
    images = image_paths(dataset_dir, split)
    if max_images > 0:
        images = images[:max_images]

    written = 0
    skipped = 0
    boxes_written = 0

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        results = model.predict(
            source=[str(path) for path in batch],
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            classes=list(COCO_TO_LOCAL),
            device=device,
            verbose=False,
        )

        for image_path, result in zip(batch, results):
            label_path = label_path_for(dataset_dir, split, image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)

            if (
                label_path.exists()
                and label_path.stat().st_size > 0
                and not overwrite
            ):
                skipped += 1
                continue

            lines: list[str] = []
            if result.boxes is not None:
                classes = result.boxes.cls.int().cpu().tolist()
                boxes = result.boxes.xywhn.cpu().tolist()
                for coco_class_id, box in zip(classes, boxes):
                    local_class_id = COCO_TO_LOCAL.get(coco_class_id)
                    if local_class_id is None:
                        continue
                    x_center, y_center, width, height = box
                    lines.append(
                        f"{local_class_id} "
                        f"{x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}"
                    )

            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            written += 1
            boxes_written += len(lines)

            if preview_dir is not None:
                preview_path = preview_dir / split / image_path.name
                write_preview(result, preview_path)

    return written, skipped, boxes_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isi label YOLO dataset dengan pseudo-label dari model COCO."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/yolo_nexar_frames"),
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa label yang sudah berisi anotasi.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Opsional: simpan gambar preview bbox.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Batasi jumlah gambar per split untuk tes cepat.",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    total_written = 0
    total_skipped = 0
    total_boxes = 0

    for split in args.splits:
        written, skipped, boxes_written = autolabel_split(
            model=model,
            dataset_dir=args.dataset_dir,
            split=split,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            overwrite=args.overwrite,
            preview_dir=args.preview_dir,
            max_images=args.max_images,
        )
        total_written += written
        total_skipped += skipped
        total_boxes += boxes_written
        print(
            f"{split}: label ditulis {written}, dilewati {skipped}, "
            f"bbox {boxes_written}"
        )

    print(f"Total label ditulis : {total_written}")
    print(f"Total label dilewati: {total_skipped}")
    print(f"Total bbox          : {total_boxes}")


if __name__ == "__main__":
    main()
