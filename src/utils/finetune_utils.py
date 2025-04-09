import os
import random

import numpy as np
import torch


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    #
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_p(dirs):
    """
    make a directory (dir) if it doesn't exist
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    return True, 'OK'
