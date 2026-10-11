
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def generate_metrics(model, loader, device, num_classes):
    model.eval()

    probas = []
    y_correto = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            prob = torch.softmax(logits, dim=1)

            probas.append(prob.cpu())
            y_correto.append(labels.cpu())

    if not y_correto:
        raise ValueError("O DataLoader está vazio.")

    y_correto = torch.cat(y_correto).numpy()
    probas = torch.cat(probas).numpy()
    y_pred = np.argmax(probas, axis=1)

    classes = list(range(num_classes))
    nomes_classes = [str(i) for i in classes]

    report = classification_report(
        y_correto,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    mcc = matthews_corrcoef(y_correto, y_pred)

    try:
        auc_ovr = roc_auc_score(
            y_correto,
            probas,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc_ovr = np.nan

    cm = confusion_matrix(
        y_correto,
        y_pred,
        labels=classes,
    )

    df_cm = pd.DataFrame(
        cm,
        index=[f"Real_{i}" for i in classes],
        columns=[f"Predito_{i}" for i in classes],
    )
    df_cm.index.name = "Classe_Real"

    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "mcc": mcc,
        "auc_roc_ovr": auc_ovr,
    }

    for classe in nomes_classes:
        metrics[f"precision_{classe}"] = report[classe]["precision"]
        metrics[f"recall_{classe}"] = report[classe]["recall"]
        metrics[f"f1_{classe}"] = report[classe]["f1-score"]

    return metrics, df_cm
