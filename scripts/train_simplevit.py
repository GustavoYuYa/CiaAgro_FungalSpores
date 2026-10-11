import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import numpy as np
import os
import pandas as pd

from src.models.simplevit import SimpleViT
from src.datasets.spore_dataset import importar_base
from src.utils.scheduler import get_scheduler
from src.utils.seed import set_seed
from src.utils.compute_statistics import calcular_mean_std
from src.datasets.transforms import transform_wout_normalize
from src.datasets.transforms import base_transform
from src.training.trainer import train_model
from src.datasets.transforms import augmentation_transform
from src.datasets.spore_dataset import importar_baseNoise
from src.datasets.transforms import noise_transform_train
from src.datasets.spore_dataset import importar_baseMixed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = r"C:\GitHub_CiaAgro\dataset_huggingface"

set_seed(42)

def load_config():

    parser = argparse.ArgumentParser(
        description="Train SimpleViT"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():

    config = load_config()

    experiment_dir = os.path.join(
        "experiments",
        "simplevit"
    )

    experiment_dir = os.path.join(
        experiment_dir,
        config["experiment"]["name"]
    )

    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    seed = config["seed"]

    set_seed(seed)

    model = SimpleViT(
        image_size=model_config["image_size"],
        patch_size=model_config["patch_size"],
        num_classes=model_config["num_classes"],
        dim=model_config["dim"],
        depth=model_config["depth"],
        heads=model_config["heads"],
        mlp_dim=model_config["mlp_dim"],
        dim_head=model_config["dim_head"]
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    scheduler = get_scheduler(
        optimizer,
        training_config["warmup_epochs"],
        training_config["epochs"]
    )

    criterion = nn.CrossEntropyLoss()

    condition = data_config["condition"]

    best_model_state = None
    best_val_loss = None
    history = None

    # images without segmentation

    if condition == "normal":

        train_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_files.npy", allow_pickle=True)
        train_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_labels.npy", allow_pickle=True)
        val_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_files.npy", allow_pickle=True)
        val_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_labels.npy", allow_pickle=True)

        _, _, _, stats_loader = importar_base(train_files, train_labels, None, None, None, None, 
                                              dataset_root=DATASET_ROOT, train_transform=transform_wout_normalize(), test_transform=None, stats=True)

        mean, std = calcular_mean_std(stats_loader)

        config["normalization"]["method"] = "calculated from training data"
        config["normalization"]["mean"] = mean.tolist()
        config["normalization"]["std"] = std.tolist()

        # M1: baseline model

        if not data_config["weighted_sampler"] and not data_config["augmentation"]:

            train_loader, val_loader, _, _ = importar_base(train_files, train_labels, val_files, val_labels, None, None, 
                                                           dataset_root=DATASET_ROOT, train_transform=base_transform(mean, std), test_transform=base_transform(mean, std))

        # M2: model with weighted sampler + data augmentation

        else:
            # with weightedramdonsampler e data augmentation
            train_loader, val_loader, _, _ = importar_base(train_files, train_labels, val_files, val_labels, None, None, 
                                                           dataset_root=DATASET_ROOT, train_transform=augmentation_transform(mean, std), test_transform=base_transform(mean, std),
                                                           stats=False, apply_weighted_sampler=True)

    # M3: model with weighted sampler + data augmentation + segmented images and noise background

    elif condition == "segmented_noise":

        train_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\train_files.npy", allow_pickle=True)
        train_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\train_labels.npy", allow_pickle=True)
        val_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\val_files.npy", allow_pickle=True)
        val_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\val_labels.npy", allow_pickle=True)

        _, _, _, stats_loader = importar_baseNoise(train_files, train_labels, None, None, None, None, 
                                                   dataset_root=DATASET_ROOT, train_transform=transform_wout_normalize(), test_transform=None, stats=True)

        mean, std = calcular_mean_std(stats_loader)

        config["normalization"]["method"] = "calculated from segmetend images + random Gaussian noise stochastic background"    
        config["normalization"]["mean"] = mean.tolist()
        config["normalization"]["std"] = std.tolist()

        train_loader, val_loader, _, _ = importar_baseNoise(train_files, train_labels, val_files, val_labels, None, None,
                                                            dataset_root=DATASET_ROOT, train_transform=noise_transform_train(), test_transform=transform_wout_normalize(),
                                                            stats=False, mean=mean, std=std)

    # M4: model with weighted sampler + data augmentation + mixed dataset (normal images and segmented noise images)

    elif condition == "mixed":

        train_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_files.npy", allow_pickle=True)
        train_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_labels.npy", allow_pickle=True)
        val_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_files.npy", allow_pickle=True)
        val_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_labels.npy", allow_pickle=True)        

        train_files_seg = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\train_files.npy", allow_pickle=True)
        val_files_seg = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\segmented_black\val_files.npy", allow_pickle=True)

        _, _, stats_loader = importar_baseMixed(train_files, train_files_seg, train_labels, val_files, val_files_seg, val_labels,
                                              dataset_root=DATASET_ROOT, train_transform_clean=transform_wout_normalize(), train_transform_seg=transform_wout_normalize(),
                                              seed=seed, test_transform_clean=None, test_transform_seg=None, stats=True)

        mean, std = calcular_mean_std(stats_loader)

        config["normalization"]["method"] = "calculated from normal images and segmented images + random Gaussian noise stochastic background"
        config["normalization"]["mean"] = mean.tolist()
        config["normalization"]["std"] = std.tolist()

        train_loader, val_loader, _ = importar_baseMixed(train_files, train_files_seg, train_labels, val_files, val_files_seg, val_labels,
                                                         dataset_root=DATASET_ROOT, train_transform_clean=augmentation_transform(mean, std), train_transform_seg=noise_transform_train(),
                                                         test_transform_clean=base_transform(mean, std), test_transform_seg=transform_wout_normalize(), 
                                                         seed=seed, stats=False, mean=mean, std=std)

    else:

        raise ValueError(f"Invalid data condition: {condition!r}")

    best_model_state, best_val_loss, best_epoch, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, training_config["epochs"], DEVICE)

    if best_model_state is None:
        raise RuntimeError("Training was not performed")

    checkpoint = {
        "model_state_dict": best_model_state,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "config": config
    }

    torch.save(
        checkpoint,
        os.path.join(experiment_dir, "checkpoint_best.pt")
    )

    history_df = pd.DataFrame(history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    history_df.to_csv(
        os.path.join(experiment_dir, "history.csv"),
        index=False
    )

if __name__ == "__main__":
    main()
