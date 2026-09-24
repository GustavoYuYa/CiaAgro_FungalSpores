import torch

def calcular_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        # images shape: (B, C, H, W)
        batch_samples = images.size(0)
        channels = images.size(1)
        height = images.size(2)
        width = images.size(3)

        total_pixels += batch_samples * height * width

        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])

    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean ** 2)

    return mean, std
