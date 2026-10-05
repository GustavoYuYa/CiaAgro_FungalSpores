from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader
import torch
import os
from sklearn.model_selection import train_test_split


class EsporosDataset(Dataset):
    def __init__(self, arquivos, labels, transform=None):
        self.arquivos = arquivos
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        path = self.arquivos[idx]
        label = self.labels[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def criar_splits():
    caminho = r"D:\CiaAgro2026\dataset_huggingface\filtered_cropped_images\Normal"

    arquivos = []
    labels = []

    for i, pasta in enumerate(sorted(os.listdir(caminho))):

        pasta_path = os.path.join(caminho, pasta)

        for subpasta in sorted(os.listdir(pasta_path)):

                """
                if pasta in "puccinias":
                    continue
                """

                subpasta_path = os.path.join(pasta_path, subpasta)

                for arquivo in os.listdir(subpasta_path):
                    arquivos.append(os.path.join(subpasta_path, arquivo))
                    labels.append(i)

    train_files, test_files, train_labels, test_labels = train_test_split(
        arquivos, labels,
        stratify=labels,
        test_size=0.20,
        random_state=42
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels,
        stratify=train_labels,
        test_size=0.0625,
        random_state=42
    )

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def importar_base(train_files, train_labels, val_files, val_labels, test_files, test_labels, transform=None):

    """
    if mean is not None:
        mean=mean.view(3, 1, 1)
        std=std.view(3, 1, 1)
    """

    train_dataset = EsporosDataset(train_files, train_labels, transform =transform)
    val_dataset   = EsporosDataset(val_files, val_labels, transform=transform)
    test_dataset  = EsporosDataset(test_files, test_labels, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True    
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    stats_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, stats_loader