import numpy as np
from torch.utils.data import WeightedRandomSampler

def weighted_random_sampler(train_labels):

    class_counts = np.bincount(train_labels)
    max_count = class_counts.max()
    target_ratio = 4 
    weights_per_class = []
    
    for c in class_counts:
        ratio = max_count / c
        ratio = min(ratio, target_ratio) 
        weights_per_class.append(ratio)

    sample_weights = [weights_per_class[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    return sampler
