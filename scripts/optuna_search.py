import optuna
import torch
import torch.optim as optim
import numpy as np
import yaml

from src.models.simplevit import SimpleViT
from src.datasets.spore_dataset import importar_base
from src.utils.seed import set_seed
from src.utils.scheduler import get_scheduler
from src.utils.compute_statistics import calcular_mean_std
from src.datasets.transforms import transform_wout_normalize
from src.datasets.transforms import base_transform
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_ROOT = r"C:\GitHub_CiaAgro\dataset_huggingface"

"""
train_files = np.load(r"C:\Analise_Esporos_2026\Modelo_Classificacao_4_classes\FilesLabels_Filtrado\train_files_Filtrado.npy", allow_pickle=True).tolist()
val_files = np.load(r"C:\Analise_Esporos_2026\Modelo_Classificacao_4_classes\FilesLabels_Filtrado\val_files_Filtrado.npy", allow_pickle=True).tolist()
train_labels = np.load(r"C:\Analise_Esporos_2026\Modelo_Classificacao_4_classes\FilesLabels_Filtrado\train_labels_Filtrado.npy", allow_pickle=True).tolist()
val_labels = np.load(r"C:\Analise_Esporos_2026\Modelo_Classificacao_4_classes\FilesLabels_Filtrado\val_labels_Filtrado.npy", allow_pickle=True).tolist()
"""

train_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_files.npy", allow_pickle=True).tolist()
train_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\train_labels.npy", allow_pickle=True).tolist()
val_files = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_files.npy", allow_pickle=True).tolist()
val_labels = np.load(r"C:\GitHub_CiaAgro\dataset_huggingface\splits\normal\val_labels.npy", allow_pickle=True).tolist()

_, _, _, stats_loader = importar_base(train_files, train_labels, None, None, None, None, DATASET_ROOT, transform_wout_normalize())

mean, std = calcular_mean_std(stats_loader)

# --------------------------------------------------
# Objective function
# --------------------------------------------------

def objective(trial):

    # Hyperparameters
    patch_size = trial.suggest_categorical(
        "patch_size",
        [4, 8]
    )

    depth = trial.suggest_int(
        "depth",
        3,
        8
    )

    heads = trial.suggest_categorical(
        "heads",
        [4, 6, 8, 12, 16]
    )

    dim_head = trial.suggest_categorical(
        "dim_head",
        [16, 32, 64]
    )

    mlp_dim = trial.suggest_categorical(
        "mlp_dim",
        [256, 384, 512, 1024]
    )

    dim = heads * dim_head

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-3,
        log=True
    )

    # Seed
    set_seed(trial.number)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_loader, val_loader, _, _ = importar_base(train_files, train_labels, val_files, val_labels, None, None, DATASET_ROOT, base_transform(mean, std))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SimpleViT(
        image_size=IMAGE_SIZE,
        patch_size=patch_size,
        num_classes=NUM_CLASSES,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head
    ).to(DEVICE)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

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

    best_model_state, best_val_loss, best_epochs, history = train_model(
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

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # Create / load Optuna study
    # --------------------------------------------------

    study = optuna.create_study(
        direction="minimize",
        study_name="simplevit_3class",
        storage="sqlite:///optuna_simplevit.db",
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
            "name": "simplevit_3class",
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
        },

        "model": {
            "patch_size": best_params["patch_size"],
            "depth": best_params["depth"],
            "heads": best_params["heads"],
            "dim_head": best_params["dim_head"],
            "dim": (
                best_params["heads"]
                * best_params["dim_head"]
            ),
            "mlp_dim": best_params["mlp_dim"],
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
            "study_name": "simplevit_3class",
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
        "configs/best_simplevit.yaml",
        "w"
    ) as f:

        yaml.dump(
            best_config,
            f,
            sort_keys=False
        )
