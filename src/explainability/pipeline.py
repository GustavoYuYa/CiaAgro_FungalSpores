"""Shared evaluation orchestration for the two independent CAM entry points."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import openpyxl

import numpy as np
import pandas as pd
import torch

from src.explainability.cam import generate_cam
from src.explainability.data import PairedTestDataset, normalization_tensors
from src.explainability.metrics import deletion_auc, localization_metrics
from src.explainability.models import cam_layer, load_model
from src.explainability.reporting import export_excel, save_heatmap
from src.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(method, argv=None):
    parser = argparse.ArgumentParser(description=f"{method}: CAMs, energy, IoU and deletion AUC")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, help="Default: DATASET_ROOT/splits")
    parser.add_argument("--architectures", nargs="+", choices=["cnn", "simplevit"], default=["cnn", "simplevit"])
    parser.add_argument("--experiments", nargs="+", choices=["M1", "M2", "M3", "M4"], default=["M1", "M2", "M3", "M4"])
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "xai")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--target", choices=["true", "predicted"], default="true", help="Notebook default: true class")
    parser.add_argument("--iou-percentile", type=float, default=90.0)
    parser.add_argument("--deletion-steps", type=int, default=100)
    parser.add_argument("--deletion-batch-size", type=int, default=32)
    parser.add_argument("--deletion-baseline", choices=["mean", "black", "gaussian"], default="mean")
    parser.add_argument("--max-samples", type=int, help="First N test indices; exploratory run only")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 <= args.iou_percentile <= 100:
        parser.error("--iou-percentile must be in [0, 100]")
    if min(args.deletion_steps, args.deletion_batch_size) < 1:
        parser.error("Deletion steps and batch size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("--run-name must contain only letters, digits, '_' or '-'")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if len(set(args.architectures)) != len(args.architectures) or len(set(args.experiments)) != len(args.experiments):
        parser.error("Architectures and experiments must not be repeated")
    return args


def evaluate_model(method, architecture, experiment, dataset, args, output_dir, device):
    checkpoint_path = args.experiments_root / architecture / experiment / "checkpoint_best.pt"
    model, config = load_model(checkpoint_path, architecture, device)
    mean, std = normalization_tensors(config, device)
    if int(dataset.labels.max()) >= config["model"]["num_classes"]:
        raise ValueError("Test label is outside the checkpoint class range.")
    layer, reshape, layer_name = cam_layer(model, architecture, config)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    model_dir = output_dir / experiment
    model_dir.mkdir(parents=True)
    protocol = {
        "architecture": architecture, "method": method,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_config": config, "target_layer": layer_name,
        "target_policy": args.target, "class_specific_cam": method == "EigenGradCAM",
        "iou_percentile": args.iou_percentile,
        "iou_rule": "cam >= percentile; zero CAM selects no pixels; empty mask gives NaN",
        "mask_rule": "transformed segmented RGB tensor: max(channel) > 0, before noise/normalization",
        "noise": "RandomState(original_test_index), normal(0.5,0.2), clipped [0,1], CHW",
        "deletion_baseline": args.deletion_baseline,
        "deletion_steps": args.deletion_steps,
        "deletion_auc_rule": "trapezoid of fixed-target softmax probability over actual fraction deleted, 0..1",
        "deletion_ties": "stable row-major pixel order; all RGB channels deleted together",
        "deletion_gaussian_seed": "original test index (same replacement for all models/conditions)",
        "overlay_reference": "original Clean image, including CAMs computed on Noise",
        "seed": args.seed, "sample_count_per_condition": limit,
        "full_test_count": len(dataset), "selection": "all" if limit == len(dataset) else "first N (exploratory)",
        "dataset_root": str(args.dataset_root.resolve()),
        "splits_root": str((args.splits_root or args.dataset_root / "splits").resolve()),
        "class_names": dataset.class_names, "device": str(device),
        "versions": {name: version(name) for name in ("torch", "torchvision", "grad-cam", "numpy", "pandas", "openpyxl")},
        "status": "running",
    }
    protocol_path = model_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    records = []
    for condition in ("Clean", "Noise"):
        condition_dir = model_dir / condition
        condition_dir.mkdir()
        with (condition_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file, \
                (condition_dir / "deletion_curves.csv").open("w", newline="", encoding="utf-8") as curves_file:
            metrics_writer = None
            curves_writer = csv.writer(curves_file)
            curves_writer.writerow(["sample_index", "target_class", "fraction_deleted", "target_probability"])
            for index in range(limit):
                sample = dataset[index]
                rgb = sample[condition].to(device)
                normalized = ((rgb - mean) / std).unsqueeze(0)
                with torch.no_grad():
                    probabilities = model(normalized).softmax(dim=1)[0]
                predicted = int(probabilities.argmax().item())
                target = sample["label"] if args.target == "true" else predicted
                heatmap = generate_cam(model, layer, reshape, normalized, target, method)
                localization = localization_metrics(heatmap, sample["mask"], args.iou_percentile)
                auc, fractions, scores = deletion_auc(
                    model, rgb, heatmap, target, mean, std,
                    steps=args.deletion_steps, batch_size=args.deletion_batch_size,
                    baseline=args.deletion_baseline, seed=index,
                )
                prefix = f"{index:06d}_{Path(sample['clean_path']).stem}"
                image_path = condition_dir / "heatmaps" / f"{prefix}.png"
                save_heatmap(
                    image_path, sample["Clean"], heatmap,
                    f"{architecture} {experiment} | {method} | {condition} | target={target} ({args.target})",
                    f"idx={index} true={sample['label']} pred={predicted} p(target)={probabilities[target].item():.4f}",
                )
                raw_dir = condition_dir / "cams"
                raw_dir.mkdir(exist_ok=True)
                np.save(raw_dir / f"{prefix}.npy", heatmap, allow_pickle=False)
                record = {
                    "architecture": architecture, "method": method, "model": experiment,
                    "condition": condition, "sample_index": index,
                    "clean_path": sample["clean_path"], "segmented_path": sample["segmented_path"],
                    "true_class": sample["label"], "class_name": sample["class_name"],
                    "predicted_class": predicted, "correct": predicted == sample["label"],
                    "target_policy": args.target, "target_class": target,
                    "predicted_probability": float(probabilities[predicted].item()),
                    "target_probability": float(probabilities[target].item()),
                    **localization, "deletion_auc": auc,
                    "deletion_initial_probability": float(scores[0]),
                    "deletion_final_probability": float(scores[-1]),
                    "heatmap_path": str(image_path.relative_to(output_dir)),
                }
                if metrics_writer is None:
                    metrics_writer = csv.DictWriter(metrics_file, fieldnames=list(record))
                    metrics_writer.writeheader()
                metrics_writer.writerow(record)
                curves_writer.writerows((index, target, float(x), float(y)) for x, y in zip(fractions, scores))
                records.append(record)
                if (index + 1) % 25 == 0 or index + 1 == limit:
                    metrics_file.flush()
                    curves_file.flush()
                    print(f"{method} | {architecture} {experiment} | {condition}: {index + 1}/{limit}", flush=True)
    protocol["status"] = "complete"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = [
        {"model": experiment, "parameter": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
        for key, value in protocol.items()
    ]
    return records, metadata


def main(method, argv=None):
    args = parse_args(method, argv)

    set_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)
    dataset = PairedTestDataset(args.dataset_root, args.splits_root)
    for architecture in args.architectures:
        for experiment in args.experiments:
            path = args.experiments_root / architecture / experiment / "checkpoint_best.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    run_dir = args.output_root / method / args.run_name
    # An explicit new run prevents stale results and accidental overwrites.
    run_dir.mkdir(parents=True, exist_ok=False)
    for architecture in args.architectures:
        output_dir = run_dir / architecture
        records, metadata = [], []
        for experiment in args.experiments:
            rows, protocol = evaluate_model(method, architecture, experiment, dataset, args, output_dir, device)
            records.extend(rows)
            metadata.extend(protocol)
            export_excel(pd.DataFrame(records), metadata, output_dir / f"{architecture}_{method}_Metrics.xlsx")
        print(f"Workbook: {output_dir / f'{architecture}_{method}_Metrics.xlsx'}", flush=True)

