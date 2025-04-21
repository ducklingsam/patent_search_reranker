import torch
from functools import lru_cache


@lru_cache(maxsize=128)
def get_device() -> str:
    """
    Определяет доступное устройство для PyTorch.
    :return: Название устройства ('cuda', 'mps' или 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = get_device()

