import math
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # warmup linear
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            # cosine decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
