import argparse, os, json, yaml, numpy as np, pandas as pd, torch
from PIL import Image
from .model import SmallUNet
from .data import load_image
from .degradation import patchwise_degrade
from .metrics import mse,rmse,psnr,ssim,bias_variance,local_mtf_nps_error

def reverse_sample(net,y,steps=25,seed=0):
    g=torch.Generator(device=y.device).manual_seed(seed)
    x=y.clone()
    for i in reversed(range(steps)):
        t=torch.full((1,), (i+1)/steps, device=y.device)
        with torch.no_grad(): score=net(x,t)
        dt=1.0/steps
        x=x+0.12*dt*score+0.03*np.sqrt(dt)*torch.randn(x.shape,generator=g,device=x.device)
        x=x.clamp(0,1)
    return x

def main(cfg):
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load("models/lafd_unet.pt",map_location=device)
    net=SmallUNet(cfg["base_channels"]).to(device); net.load_state_dict(ck["ema"]); net.eval()
    df=pd.read_csv("data/manifests/manifest.csv"); df=df[df.split=="test"].head(10)
    os.makedirs("results/inference",exist_ok=True); rows=[]
    for j, row in df.iterrows():
        gt=load_image(row.path,cfg["image_size"])[0]
        y,_=patchwise_degrade(gt,patch_size=cfg["patch_size"],stride=cfg["patch_stride"],seed=cfg["seed"]+int(j))
        yy=torch.from_numpy(y[None,None]).float().to(device)
        samples=[]
        for k in range(cfg["uncertainty_runs"]):
            samples.append(reverse_sample(net,yy,cfg["reverse_steps"],cfg["seed"]+k)[0,0].cpu().numpy())
        samples=np.stack(samples)
        restored=samples.mean(0)
        bias,var=bias_variance(samples,gt)
        me,ne=local_mtf_nps_error(restored,gt,cfg["patch_size"],cfg["patch_stride"])
        rows.append({"path":row.path,"class":row["class"],"MSE":mse(restored,gt),"Bias":bias,"Variance":var,"RMSE":rmse(restored,gt),"PSNR":psnr(restored,gt),"SSIM":ssim(restored,gt),"MTF_Error":me,"NPS_Error":ne})
        stem=f"{j}_{row['class']}"
        Image.fromarray((gt*255).astype(np.uint8)).save(f"results/inference/{stem}_ground_truth.png")
        Image.fromarray((y*255).astype(np.uint8)).save(f"results/inference/{stem}_degraded.png")
        Image.fromarray((restored*255).astype(np.uint8)).save(f"results/inference/{stem}_restored.png")
        Image.fromarray((np.sqrt(np.maximum(samples.var(0),0))*255/(np.sqrt(np.maximum(samples.var(0),0)).max()+1e-8)).astype(np.uint8)).save(f"results/inference/{stem}_uncertainty.png")
    pd.DataFrame(rows).to_csv("results/inference/metrics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("SAVED results/inference/metrics.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); args=ap.parse_args()
    with open(args.config) as f: cfg=yaml.safe_load(f)
    main(cfg)
