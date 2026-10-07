import argparse, os, copy, yaml, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import RestorationDataset
from .model import SmallUNet
from .utils import seed_everything

def run(cfg):
    seed_everything(cfg["seed"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    ds=RestorationDataset("data/manifests/manifest.csv","train",cfg["image_size"],cfg["seed"])
    vs=RestorationDataset("data/manifests/manifest.csv","val",cfg["image_size"],cfg["seed"]+10000)
    dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,num_workers=0)
    vl=DataLoader(vs,batch_size=cfg["batch_size"],shuffle=False,num_workers=0)
    net=SmallUNet(cfg["base_channels"]).to(device)
    ema=copy.deepcopy(net).eval()
    opt=torch.optim.Adam(net.parameters(),lr=cfg["learning_rate"])
    best=1e9
    hist=[]
    for epoch in range(cfg["epochs"]):
        net.train(); losses=[]
        for y,x,_ in tqdm(dl,desc=f"epoch {epoch+1}/{cfg['epochs']}"):
            y,x=y.to(device),x.to(device)
            t=torch.rand(y.shape[0],device=device)
            alpha=(1-0.7*t).sqrt()[:,None,None,None]
            sigma=(0.05+0.25*t)[:,None,None,None]
            eps=torch.randn_like(x)
            xt=alpha*x+(1-alpha)*y+sigma*eps
            target=-eps/sigma
            pred=net(xt,t)
            loss=((pred-target)**2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            with torch.no_grad():
                for p,ep in zip(net.parameters(),ema.parameters()):
                    ep.mul_(0.999).add_(p,alpha=0.001)
            losses.append(float(loss))
        net.eval(); val=[]
        with torch.no_grad():
            for y,x,_ in vl:
                y,x=y.to(device),x.to(device); t=torch.full((y.shape[0],),0.5,device=device)
                pred=net(y,t); val.append(float(pred.abs().mean()))
        row={"epoch":epoch+1,"train_loss":sum(losses)/len(losses),"val_score":sum(val)/len(val)}
        hist.append(row); print(row)
        if row["train_loss"]<best:
            best=row["train_loss"]
            os.makedirs("models",exist_ok=True)
            torch.save({"model":net.state_dict(),"ema":ema.state_dict(),"config":cfg},"models/lafd_unet.pt")
    os.makedirs("results/training",exist_ok=True)
    with open("results/training/history.json","w") as f: json.dump(hist,f,indent=2)
    print("DEVICE",device)
    print("SAVED models/lafd_unet.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml")
    args=ap.parse_args()
    with open(args.config) as f: cfg=yaml.safe_load(f)
    run(cfg)
