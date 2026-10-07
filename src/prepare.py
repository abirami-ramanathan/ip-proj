import argparse, yaml, os
from .data import make_manifest
from .utils import seed_everything
ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args()
with open(a.config) as f: cfg=yaml.safe_load(f)
seed_everything(cfg["seed"])
df=make_manifest(a.dataset,"data/manifests/manifest.csv",cfg["images_per_class"],cfg["seed"])
print(df.groupby(["split","class"]).size())
print("TOTAL SELECTED",len(df))
print("MANIFEST data/manifests/manifest.csv")
