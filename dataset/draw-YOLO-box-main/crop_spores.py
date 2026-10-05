#!/usr/bin/env python3
# coding: utf-8
"""Extract YOLO-format bounding boxes from images and save them as cropped images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Caminhos padrão
DEFAULT_LABEL_FOLDER = "./labels/"
DEFAULT_RAW_IMAGE_FOLDER = "./raw_images/"
DEFAULT_CLASS_PATH = "./classes.txt"
DEFAULT_IMAGE_NAME_LIST_PATH = "./name_list.txt"

# Diretório de saída exigido
DEFAULT_OUTPUT_CROP_FOLDER = r"C:\GitHub_CiaAgro\CiaAgro_FungalSpores\data\cropped_images"

logger = logging.getLogger(__name__)


def find_image_path(raw_image_folder: str, stem: str) -> Optional[Path]:
    folder = Path(raw_image_folder)
    for ext in sorted(IMAGE_EXTENSIONS):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    if folder.is_dir():
        lower_stem = stem.lower()
        for path in folder.iterdir():
            if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
            if path.is_file() and path.stem.lower() == lower_stem and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
    return None


def load_classes(class_path: str) -> List[str]:
    path = Path(class_path)
    if not path.is_file():
        raise FileNotFoundError(f"classes file not found: {class_path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"classes file is empty: {class_path}")
    return classes


def yolo_line_to_xyxy(parts: Sequence[str], width: int, height: int) -> Tuple[int, List[int]]:
    class_idx = int(parts[0])
    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_w = float(parts[3]) * width
    box_h = float(parts[4]) * height
    x1 = int(round(x_center - box_w / 2))
    y1 = int(round(y_center - box_h / 2))
    x2 = int(round(x_center + box_w / 2))
    y2 = int(round(y_center + box_h / 2))
    
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    return class_idx, [x1, y1, x2, y2]


def crop_boxes_from_image(
    image_stem: str,
    classes: Sequence[str],
    label_folder: str,
    raw_image_folder: str,
    output_crop_folder: str,
    class_counters: Dict[int, int],  # Dicionário adicionado para rastrear a numeração global
) -> int:
    if image_stem.startswith("."):
        return 0

    image_path = find_image_path(raw_image_folder, image_stem)
    if image_path is None:
        logger.warning("Imagem não encontrada para %r em %s", image_stem, raw_image_folder)
        return 0

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Falha ao ler a imagem: %s", image_path)
        return 0

    height, width = image.shape[:2]
    label_path = Path(label_folder) / f"{image_stem}.txt"
    box_number = 0
    
    base_out_dir = Path(output_crop_folder)

    if label_path.is_file():
        with label_path.open("r", encoding="utf-8") as source_file:
            for line_no, line in enumerate(source_file, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_idx, xyxy = yolo_line_to_xyxy(parts, width, height)
                except ValueError:
                    continue
                if class_idx < 0 or class_idx >= len(classes):
                    continue
                
                x1, y1, x2, y2 = xyxy
                
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_spore = image[y1:y2, x1:x2]

                class_name = classes[class_idx]
                safe_class_name = class_name.replace("/", "_").replace("\\", "_")
                class_dir = base_out_dir / safe_class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                ext = image_path.suffix.lower()
                
                # Pega o número atual do contador para esta classe e formata com 5 zeros (00001)
                current_count = class_counters[class_idx]
                crop_filename = f"{safe_class_name}_{current_count:05d}{ext}"
                crop_path = class_dir / crop_filename
                
                if not cv2.imwrite(str(crop_path), cropped_spore):
                    logger.error("Falha ao salvar o recorte %s", crop_path)
                else:
                    # Só incrementa se salvou com sucesso
                    class_counters[class_idx] += 1
                    box_number += 1
    
    return box_number


def make_name_list(raw_image_folder: str, image_name_list_path: str) -> List[str]:
    folder = Path(raw_image_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"raw image folder not found: {raw_image_folder}")

    stems: List[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stems.append(path.stem)

    list_path = Path(image_name_list_path)
    with list_path.open("w", encoding="utf-8") as handle:
        for stem in stems:
            handle.write(stem + "\n")
    return stems


def read_name_list(image_name_list_path: str) -> List[str]:
    path = Path(image_name_list_path)
    if not path.is_file():
        raise FileNotFoundError(f"name list not found: {image_name_list_path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.read().split("\n") if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop YOLO boxes from images and sort by class.")
    parser.add_argument("--labels", default=DEFAULT_LABEL_FOLDER, help="YOLO txt label folder")
    parser.add_argument("--images", default=DEFAULT_RAW_IMAGE_FOLDER, help="Raw image folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CROP_FOLDER, help="Output cropped images folder")
    parser.add_argument("--classes", default=DEFAULT_CLASS_PATH, help="Class names file")
    parser.add_argument("--name-list", default=DEFAULT_IMAGE_NAME_LIST_PATH, help="Path to write/read image stem list")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    classes = load_classes(args.classes)
    stems = make_name_list(args.images, args.name_list)
    stems = read_name_list(args.name_list)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Inicia um contador (começando no 1) para cada classe existente no arquivo classes.txt
    # Ex: {0: 1, 1: 1, 2: 1}
    class_counters = {i: 1 for i in range(len(classes))}

    crop_total = 0
    image_total = 0
    
    logger.info("Iniciando o recorte e extração dos esporos...")
    
    for stem in stems:
        crop_num = crop_boxes_from_image(
            stem,
            classes,
            args.labels,
            args.images,
            args.output,
            class_counters  # Passando o dicionário de contadores
        )
        crop_total += crop_num
        if crop_num > 0:
            image_total += 1

    logger.info("CONCLUÍDO: %d imagens-fonte processadas, %d recortes extraídos.", image_total, crop_total)
    
    # Imprime um resumo de quantas imagens foram geradas por classe
    logger.info("--- Resumo por Classe ---")
    for i, class_name in enumerate(classes):
        total_class = class_counters[i] - 1
        logger.info(f"{class_name}: {total_class} imagens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())