from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import CLASSES

def discover_images(dataset_root):
    root = Path(dataset_root)
    paths=[]
    for class_dir, label in CLASSES.items():
        candidates=list((root/"lung_image_sets"/class_dir).glob("*.jpeg"))+list((root/"lung_image_sets"/class_dir).glob("*.jpg"))
        if not candidates:
            candidates=list((root/"colon_image_sets"/class_dir).glob("*.jpeg"))+list((root/"colon_image_sets"/class_dir).glob("*.jpg"))
        for p in candidates:
            paths.append((p, class_dir, label))
    return paths

def make_manifest(dataset_root, out_csv, per_class=200, seed=42):
    all_items=discover_images(dataset_root)
    rows=[]
    rng=np.random.default_rng(seed)
    for class_dir,label in CLASSES.items():
        group=[x for x in all_items if x[1]==class_dir]
        if len(group)<per_class:
            raise RuntimeError(f"{class_dir}: found {len(group)} images, need {per_class}")
        idx=rng.choice(len(group), size=per_class, replace=False)
        for i in idx:
            p,cd,lbl=group[i]
            rows.append({"path":str(p), "class":cd, "label":lbl})
    df=pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    train, temp=train_test_split(df, test_size=0.2, stratify=df["class"], random_state=seed)
    val, test=train_test_split(temp, test_size=0.5, stratify=temp["class"], random_state=seed)
    train=train.assign(split="train"); val=val.assign(split="val"); test=test.assign(split="test")
    out=pd.concat([train,val,test], ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv,index=False)
    return out

def load_image(path, size=256):
    im=Image.open(path).convert("L").resize((size,size), Image.Resampling.BILINEAR)
    arr=np.asarray(im).astype(np.float32)/255.0
    arr=np.clip(arr,0,1)
    return arr[None,...]
