import pandas as pd
import torch
from torch.utils.data import Dataset
from .data import load_image
from .degradation import patchwise_degrade
class RestorationDataset(Dataset):
    def __init__(self, manifest_csv, split, size=256, seed=42):
        self.df=pd.read_csv(manifest_csv)
        self.df=self.df[self.df.split==split].reset_index(drop=True)
        self.size=size; self.seed=seed
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        x=load_image(row.path,self.size)[0]
        y,_=patchwise_degrade(x, seed=self.seed+idx)
        return torch.from_numpy(y[None]), torch.from_numpy(x[None]), int(idx)
