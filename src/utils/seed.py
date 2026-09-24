import random
import numpy as np
import os
import torch


def set_seed(seed=42):
    # 1. Trava o Python e o NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Trava o PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 3. Trava o comportamento da placa de vídeo (O segredo da reprodutibilidade)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    