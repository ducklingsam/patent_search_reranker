import torch
from functools import lru_cache
import os
import requests
from tqdm import tqdm


@lru_cache(maxsize=128)
def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = get_device()


def download_model(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192

    with open(output_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

