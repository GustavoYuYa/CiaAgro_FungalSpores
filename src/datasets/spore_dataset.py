from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from pathlib import Path
# import os
# from sklearn.model_selection import train_test_split

from src.utils.sampler import weighted_random_sampler
from src.datasets.transforms import masked_jitter

# Normal images


class EsporosDataset(Dataset):
    def __init__(self, arquivos, labels, dataset_root, transform=None):
        self.arquivos = arquivos
        self.labels = labels
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        path = Path(self.dataset_root) / Path(self.arquivos[idx])

        label = self.labels[idx]

        img = cv2.imread(str(path))

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def importar_base(train_files=None, train_labels=None, val_files=None, val_labels=None, test_files=None, test_labels=None,  
                  dataset_root=None, train_transform=None, test_transform=None, stats=False, apply_weighted_sampler=False):

    # Training

    if train_files is not None:
        train_dataset = EsporosDataset(train_files, train_labels, dataset_root, transform=train_transform)

        if stats:
            stats_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=False
            )

            train_loader = None

        else:
            stats_loader = None

            if apply_weighted_sampler:
                sampler = weighted_random_sampler(train_labels)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=64,
                    sampler=sampler,
                    pin_memory=True    
                )

            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=64,
                    shuffle=True,
                    pin_memory=True
                )

    else:
        train_loader = None
        stats_loader = None

    # Validation

    if val_files is not None:
        val_dataset = EsporosDataset(val_files, val_labels, dataset_root, transform=test_transform)

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False
        )

    else:
        val_loader = None

    # Test
    
    if test_files is not None:
        test_dataset  = EsporosDataset(test_files, test_labels, dataset_root, transform=test_transform)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

    else:
        test_loader = None

    return train_loader, val_loader, test_loader, stats_loader


# Noise images


class EsporosDatasetNoise(Dataset):
    def __init__(self, arquivos, labels, dataset_root, transform=None,
                 deterministic_noise=False, apply_masked_jitter=False, mean=None, std=None):
        self.arquivos = arquivos
        self.labels = labels

        self.dataset_root = dataset_root

        self.transform = transform

        self.deterministic_noise = deterministic_noise
        self.apply_masked_jitter = apply_masked_jitter

        if mean is not None and std is not None:
            self.mean = mean.view(3, 1, 1)
            self.std = std.view(3, 1, 1)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        path = Path(self.dataset_root) / Path(self.arquivos[idx])

        label = self.labels[idx]

        img = cv2.imread(str(path))

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_tensor = self.transform(img)
        
        mask = (img_tensor.max(dim=0, keepdims=True).values > 0).float()

        rng_local = np.random if not self.deterministic_noise else np.random.RandomState(idx)

        noise = rng_local.normal(loc=0.5, scale=0.2, size=tuple(img_tensor.shape))

        noise = np.clip(noise, 0, 1).astype(np.float32)
        noise_tensor = torch.from_numpy(noise)
        img_tensor = (img_tensor * mask) + (noise_tensor * (1 - mask))
        
        if self.apply_masked_jitter:
            img_tensor = masked_jitter(
            img_tensor,
            mask
        )
        
        if self.mean is not None and self.std is not None:
            img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor, label


def importar_baseNoise(train_files=None, train_labels=None, val_files=None, val_labels=None, test_files=None, test_labels=None,  
                  dataset_root=None, train_transform=None, test_transform=None, stats=False, mean=None, std=None):

    # Training

    if train_files is not None:

        if stats:
            train_dataset = EsporosDatasetNoise(train_files, train_labels, dataset_root, train_transform, True) 

            stats_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=False
            )

            train_loader = None

        else:
            stats_loader = None

            train_dataset = EsporosDatasetNoise(train_files, train_labels, dataset_root, transform=train_transform, 
                                                deterministic_noise=False, apply_masked_jitter=True, mean=mean, std=std) 

            sampler = weighted_random_sampler(train_labels)

            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                sampler=sampler,
                pin_memory=True    
            )

    else:
        train_loader = None
        stats_loader = None

    # Validation

    if val_files is not None:
        val_dataset = EsporosDatasetNoise(val_files, val_labels, dataset_root, transform=test_transform,
                                          deterministic_noise=True, apply_masked_jitter=False, mean=mean, std=std)

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False
        )

    else:
        val_loader = None

    # Test
    
    if test_files is not None:
        test_dataset  = EsporosDatasetNoise(test_files, test_labels, dataset_root, transform=test_transform,
                                            deterministic_noise=True, apply_masked_jitter=False, mean=mean, std=std)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

    else:
        test_loader = None

    return train_loader, val_loader, test_loader, stats_loader


class EsporosDatasetMixed(Dataset):

    def __init__(self, clean_dataset, noise_dataset, training=True, seed=42):
        if len(clean_dataset) != len(noise_dataset):
            raise ValueError(
                "Clean and noise datasets must have the same length."
            )

        self.clean_dataset = clean_dataset
        self.noise_dataset = noise_dataset
        self.training = training

        rng = np.random.RandomState(seed)

        # Fixed 50/50 assignment for validation/test
        self.use_noise = rng.rand(len(self.clean_dataset)) < 0.5

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):

        if self.training:
            use_noise = np.random.rand() < 0.5
        else:
            use_noise = self.use_noise[idx]

        if use_noise:
            return self.noise_dataset[idx]

        return self.clean_dataset[idx]


def importar_baseMixed(train_filesClean, train_filesBlack, train_labels, val_filesClean, val_filesBlack, val_labels,
                       dataset_root, train_transform_clean=None, train_transform_seg=None, test_transform_clean=None, test_transform_seg=None,
                       seed=42, stats=False, mean=None, std=None):

    if stats:

        clean_train_dataset = EsporosDataset(
            train_filesClean,
            train_labels,
            dataset_root,
            transform=train_transform_clean
        )

        noise_train_dataset = EsporosDatasetNoise(
            train_filesBlack,
            train_labels,
            dataset_root,
            transform=train_transform_seg,
            deterministic_noise=True,
            apply_masked_jitter=False,
            mean=None,
            std=None
        )

        train_dataset = EsporosDatasetMixed(
            clean_train_dataset,
            noise_train_dataset, 
            training=False,
            seed=seed
        )

        stats_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False
        )

        train_loader = None
        val_loader = None

    else:

        clean_train_dataset = EsporosDataset(
            train_filesClean,
            train_labels,
            dataset_root,
            transform=train_transform_clean
        )

        noise_train_dataset = EsporosDatasetNoise(
            train_filesBlack,
            train_labels,
            dataset_root,
            transform=train_transform_seg,
            deterministic_noise=False,
            apply_masked_jitter=True,
            mean=mean,
            std=std
        )

        train_dataset = EsporosDatasetMixed(
            clean_train_dataset,
            noise_train_dataset,
            training=True,
            seed=seed
        )

        clean_val_dataset = EsporosDataset(
            val_filesClean,
            val_labels,
            dataset_root,
            transform=test_transform_clean
        )

        noise_val_dataset = EsporosDatasetNoise(
            val_filesBlack,
            val_labels,
            dataset_root,
            transform=test_transform_seg,
            deterministic_noise=True,
            apply_masked_jitter=False,
            mean=mean,
            std=std
        )

        val_dataset = EsporosDatasetMixed(
            clean_val_dataset,
            noise_val_dataset,
            training=False,
            seed=seed
        )

        sampler = weighted_random_sampler(train_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            sampler=sampler,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False
        )

        stats_loader = None

    return train_loader, val_loader, stats_loader

# def criar_splits(caminho=None):
#     if caminho is None:
#         caminho = r"D:\CiaAgro2026\dataset_huggingface\filtered_cropped_images\Normal"

#     arquivos = []
#     labels = []

#     class_dirs = sorted(
#         [
#             pasta for pasta in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, pasta))
#         ]
#     )

#     for i, pasta in enumerate(class_dirs):

#         pasta_path = os.path.join(caminho, pasta)

#         subpastas = sorted(
#             [
#                 subpasta for subpasta in os.listdir(pasta_path) if os.path.isdir(os.path.join(pasta_path, subpasta))
#             ]
#         )

#         if subpastas:

#             for subpasta in subpastas:
#                 subpasta_path = os.path.join(pasta_path,subpasta)

#                 for arquivo in os.listdir(subpasta_path):
#                     arquivos.append(os.path.join(subpasta_path, arquivo))
#                     labels.append(i)

#         else:

#             for arquivo in os.listdir(pasta_path):
#                 arquivo_path = os.path.join(pasta_path, arquivo)

#                 if os.path.isfile(arquivo_path):
#                     arquivos.append(arquivo_path)
#                     labels.append(i)

#     train_files, test_files, train_labels, test_labels = train_test_split(
#         arquivos, labels,
#         stratify=labels,
#         test_size=0.20,
#         random_state=42
#     )

#     train_files, val_files, train_labels, val_labels = train_test_split(
#         train_files, train_labels,
#         stratify=train_labels,
#         test_size=0.0625,
#         random_state=42
#     )

#     return train_files, train_labels, val_files, val_labels, test_files, test_labels
