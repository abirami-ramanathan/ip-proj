from pathlib import Path
import random
import numpy as np
import torch

CLASSES = {
    "lung_aca": "Lung Adenocarcinoma",
    "lung_n": "Lung Benign",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "colon_aca": "Colon Adenocarcinoma",
    "colon_n": "Colon Benign",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
