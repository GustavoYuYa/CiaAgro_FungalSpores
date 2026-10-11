import optuna

import torch
import torch.optim as optim

import numpy as np
import yaml

from src.models.cnn import SimpleCNN

from src.datasets.spore_dataset import importar_base

from src.utils.seed import set_seed
from src.utils.scheduler import get_scheduler
from src.utils.compute_statistics import calcular_mean_std

from src.datasets.transforms import (transform_wout_normalize,base_transform)

from src.training.trainer import train_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLASSES = 3
IMAGE_SIZE = 64

BATCH_SIZE = 64

EPOCHS = 100
WARMUP_EPOCHS = 20

WEIGHT_DECAY = 0.05

N_TRIALS = 30

channel_options = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256)
    # "large": (64, 128, 256, 512)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = r"C:\GitHub_CiaAgro\dataset_huggingface"

train_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_files.npy", allow_pickle=True).tolist()
train_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_labels.npy", allow_pickle=True).tolist()
val_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_files.npy", allow_pickle=True).tolist()
val_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_labels.npy", allow_pickle=True).tolist()

_, _, _, stats_loader = importar_base(train_files, train_labels, None, None, None, None,
                                      dataset_root=DATASET_ROOT, train_transform=transform_wout_normalize(), test_transform=None, stats=True)

mean, std = calcular_mean_std(stats_loader)

def objective(trial):

    set_seed(trial.number)

    channels_key = trial.suggest_categorical("channels", list(channel_options.keys()))

    channels = channel_options[channels_key]

    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)

    train_loader, val_loader, _, _ = importar_base(train_files, train_labels, val_files, val_labels, None, None,
                                                   dataset_root=DATASET_ROOT, train_transform=base_transform(mean, std), test_transform=base_transform(mean, std))

    model = SimpleCNN(
        num_classes=NUM_CLASSES,
        channels=channels,
        dropout=dropout
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_scheduler(
        optimizer,
        WARMUP_EPOCHS,
        EPOCHS
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_model_state, best_val_loss, best_epoch, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE
    )

    return best_val_loss


if __name__ == "__main__":

    # --------------------------------------------------
    # Create / load Optuna study
    # --------------------------------------------------

    study = optuna.create_study(
        direction="minimize",
        study_name="cnn_3class",
        storage="sqlite:///optuna_cnn.db",
        load_if_exists=True
    )

    print(f"\nExisting trials: {len(study.trials)}")

    for trial in study.trials:
        print(
            f"Trial {trial.number}: "
            f"{trial.state.name} "
            f"{trial.value}"
        )

    # --------------------------------------------------
    # Mark interrupted trials as FAILED
    # --------------------------------------------------

    running_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]

    if running_trials:

        print("\nInterrupted trials detected:")

        for trial in running_trials:

            print(
                f"Trial {trial.number} "
                f"-> marking as FAILED"
            )

            study.tell(
                trial.number,
                state=optuna.trial.TrialState.FAIL
            )

    # --------------------------------------------------
    # Continue until we have N_TRIALS COMPLETED trials
    # --------------------------------------------------

    while True:

        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        n_completed = len(completed_trials)

        print(
            f"\nCompleted trials: "
            f"{n_completed}/{N_TRIALS}"
        )

        # Target reached
        if n_completed >= N_TRIALS:
            break

        trials_remaining = N_TRIALS - n_completed

        print(
            f"Running {trials_remaining} "
            f"more trial(s)..."
        )

        study.optimize(
            objective,
            n_trials=trials_remaining
        )

    # --------------------------------------------------
    # Get best trial
    # --------------------------------------------------

    best_trial = study.best_trial
    best_params = best_trial.params

    print("\n========================================")
    print("OPTUNA SEARCH FINISHED")
    print("========================================")

    print(
        f"Completed trials: {n_completed}"
    )

    print(
        f"Best trial: {best_trial.number}"
    )

    print(
        f"Best validation loss: "
        f"{best_trial.value:.6f}"
    )

    print("\nBest hyperparameters:")

    for key, value in best_params.items():
        print(f"{key}: {value}")

    # --------------------------------------------------
    # Create final configuration
    # --------------------------------------------------

    best_config = {

        "experiment": {
            "name": "cnn_3class",
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
        },

        "model": {
            "channels": list(channel_options[best_params["channels"]]),
            "dropout": best_params["dropout"]
        },

        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "learning_rate": best_params["learning_rate"],
        },

        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },

        "optuna": {
            "study_name": "cnn_3class",
            "n_trials_target": N_TRIALS,
            "n_trials_completed": n_completed,
            "best_trial": best_trial.number,
            "best_validation_loss": best_trial.value,
        },

        "optimizer": {
            "name": "AdamW",
            "weight_decay": WEIGHT_DECAY,
        },

        "scheduler": {
            "name": "warmup_cosine",
            "warmup_epochs": WARMUP_EPOCHS,
        }
    }

    # --------------------------------------------------
    # Save YAML
    # --------------------------------------------------

    with open(
        "configs/best_cnn.yaml",
        "w"
    ) as f:

        yaml.safe_dump(
            best_config,
            f,
            sort_keys=False
        )
