from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_images(root: Path, limit_per_folder: int = 5):
    """Return at most limit_per_folder images from each leaf class folder."""
    found = []
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files[:limit_per_folder]:
            found.append((path, folder.name))
    return found


def load_image(path: Path, size: int = 256) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def split_by_class(records, test_fraction=0.2):
    train, test = [], []
    by_class = {}
    for rec in records:
        by_class.setdefault(rec[1], []).append(rec)
    for group in by_class.values():
        n_test = max(1, int(round(len(group) * test_fraction))) if len(group) > 1 else 0
        test.extend(group[:n_test]); train.extend(group[n_test:])
    return train, test
