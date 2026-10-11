
import os
import argparse

import numpy as np
import pandas as pd
import torch

from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

from src.evaluation.metrics import generate_metrics
from src.models.simplevit import SimpleViT
from src.models.cnn import CNN
from src.datasets.spore_dataset import importar_base, importar_baseNoise
from src.datasets.transforms import (
    transform_wout_normalize,
    base_transform
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = r"C:\GitHub_CiaAgro\dataset_huggingface"
# simplevit path
# EXPERIMENTS_ROOT = os.path.join("experiments", "simplevit")
# cnn path
EXPERIMENTS_ROOT = os.path.join("experiments", "cnn")

test_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\test_files.npy", allow_pickle=True)
test_files_black = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\test_files.npy", allow_pickle=True)
test_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\test_labels.npy", allow_pickle=True)
test_labels_black = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\test_labels.npy", allow_pickle=True)


def build_model(config):

    cfg = config["model"]

    # loading simplevit model
    # return SimpleViT(
    #     image_size=cfg["image_size"],
    #     patch_size=cfg["patch_size"],
    #     num_classes=cfg["num_classes"],
    #     dim=cfg["dim"],
    #     depth=cfg["depth"],
    #     heads=cfg["heads"],
    #     mlp_dim=cfg["mlp_dim"],
    #     dim_head=cfg["dim_head"],
    # ).to(DEVICE)

    # loading cnn model
    return CNN(
        num_classes=cfg["num_classes"],
        channels=cfg["channels"],
        dropout=cfg["dropout"]
    ).to(DEVICE)


def build_test_loaders(config):

    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]

    if mean is None or std is None:
        raise ValueError(
            "O checkpoint não contém mean/std na configuração."
        )

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    # Clean Test: original images
    _, _, test_loader_clean, _ = importar_base(
        None,
        None,
        None,
        None,
        test_files=test_files,
        test_labels=test_labels,
        dataset_root=DATASET_ROOT,
        train_transform=None,
        test_transform=base_transform(mean, std)
    )

    # Noise Test: segmented images with noise background
    _, _, test_loader_noise, _ = importar_baseNoise(
        None,
        None,
        None,
        None,
        test_files=test_files_black,
        test_labels=test_labels_black,
        dataset_root=DATASET_ROOT,
        train_transform=None,
        test_transform=transform_wout_normalize(),
        stats=False,
        mean=mean,
        std=std,
    )

    return test_loader_clean, test_loader_noise


def export_excel(df_metrics, matrices_results, excel_path):
    
    rows = []

    for nome_modelo in df_metrics["model"].drop_duplicates():
        group = df_metrics[df_metrics["model"] == nome_modelo]

        rows.extend(group.to_dict("records"))
        rows.append({})

    df_excel_metrics = pd.DataFrame(
        rows,
        columns=df_metrics.columns,
    )

    dir = os.path.dirname(excel_path)
    if dir:
        os.makedirs(dir, exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        df_excel_metrics.to_excel(
            writer,
            sheet_name="Metrics",
            index=False,
        )

        ws = writer.book.create_sheet("Confusion_Matrices")

        model_names = list(matrices_results.keys())
        first = model_names[0]

        matrix_columns = [
            "Classe_Real",
            *matrices_results[first]["Clean"].columns.tolist(),
        ]

        weight = len(matrix_columns)
        col_clean = 1
        col_noise = weight + 2
        row = 1

        for model_name in model_names:
            for condition, initial_column in (
                ("Clean", col_clean),
                ("Noise", col_noise),
            ):
                df_cm = matrices_results[model_name][condition]

                ws.merge_cells(
                    start_row=row,
                    start_column=initial_column,
                    end_row=row,
                    end_column=initial_column + weight - 1,
                )

                title = ws.cell(
                    row=row,
                    column=initial_column,
                    value=f"{model_name} - {condition}",
                )
                title.font = Font(bold=True, size=12)
                title.alignment = Alignment(horizontal="center")

            row += 1

            for j, column_name in enumerate(matrix_columns):
                for start in (col_clean, col_noise):
                    cell = ws.cell(
                        row=row,
                        column=start + j,
                        value=column_name,
                    )
                    cell.font = Font(bold=True)

            row += 1

            cm_clean = matrices_results[model_name]["Clean"]
            cm_noise = matrices_results[model_name]["Noise"]

            for i in range(len(cm_clean)):
                ws.cell(row, col_clean, cm_clean.index[i])
                ws.cell(row, col_noise, cm_noise.index[i])

                for j, column_name in enumerate(cm_clean.columns, start=1):
                    ws.cell(
                        row,
                        col_clean + j,
                        int(cm_clean.iloc[i][column_name]),
                    )
                    ws.cell(
                        row,
                        col_noise + j,
                        int(cm_noise.iloc[i][column_name]),
                    )

                row += 1

            row += 1

        for column_idx in range(1, ws.max_column + 1):
            max_length = 0
            letter = get_column_letter(column_idx)

            for row_idx in range(1, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=column_idx)

                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )

            ws.column_dimensions[letter].width = min(max_length + 2, 30)

    print(f"Excel salvo em: {excel_path}")


def compute(experiment_names, excel_path):
    metrics_results = []
    matrices_results = {}

    for name in experiment_names:
        checkpoint_path = os.path.join(
            EXPERIMENTS_ROOT,
            name,
            "checkpoint_best.pt",
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

        config = checkpoint["config"]

        print(f"\nLoading experiment: {name}")

        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_loader_clean, test_loader_noise = build_test_loaders(
            config
        )

        num_classes = config["model"]["num_classes"]

        matrices_results[name] = {}

        for condition, loader in (
            ("Clean", test_loader_clean),
            ("Noise", test_loader_noise),
        ):
            metrics, df_cm = generate_metrics(
                model=model,
                loader=loader,
                device=DEVICE,
                num_classes=num_classes,
            )

            metrics_results.append({
                "model": name,
                "condition": condition,
                **metrics,
            })

            matrices_results[name][condition] = df_cm

            print(
                f"{name} | {condition} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"MCC: {metrics['mcc']:.4f}"
            )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    df_metrics = pd.DataFrame(metrics_results)

    export_excel(
        df_metrics=df_metrics,
        matrices_results=matrices_results,
        excel_path=excel_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SimpleViT checkpoints"
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Names of experiment folders",
    )

    parser.add_argument(
        "--output",
        default="Metricas_SimpleViT.xlsx",
        help="Output Excel path",
    )

    args = parser.parse_args()

    compute(
        experiment_names=args.experiments,
        excel_path=args.output,
    )
