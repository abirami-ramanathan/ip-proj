import argparse, csv, json, random, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .data import discover_images, load_image, split_by_class
from .diffusion import patchwise_forward, local_descriptors
from .model import ScoreUNet


def descriptor_maps(image,patch=64,stride=32):
    desc,pos=local_descriptors(image,patch,stride)
    maps=torch.zeros((2,*image.shape[-2:]),device=image.device); counts=torch.zeros_like(maps[:1])
    for d,(y,x) in zip(desc,pos):
        maps[:,y:y+patch,x:x+patch]+=d[:,None,None]; counts[:,y:y+patch,x:x+patch]+=1
    return maps/counts.clamp_min(1)

def augment(image,rng):
    if rng.random()<0.5: image=image.flip(-1)
    if rng.random()<0.5: image=image.flip(-2)
    k=rng.randrange(4)
    return torch.rot90(image,k,(-2,-1)).contiguous()

def gradient_loss(pred,target):
    pdx=pred[:,:,:,1:]-pred[:,:,:,:-1]; tdx=target[:,:,:,1:]-target[:,:,:,:-1]
    pdy=pred[:,:,1:,:]-pred[:,:,:-1,:]; tdy=target[:,:,1:,:]-target[:,:,:-1,:]
    return F.l1_loss(pdx,tdx)+F.l1_loss(pdy,tdy)

def save_gray(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    arr=np.uint8(np.clip(array,0,1)*255)
    if arr.ndim==3 and arr.shape[0] in (3,4): arr=np.moveaxis(arr,0,-1)
    Image.fromarray(arr).save(path)

def train(args):
    root=Path(args.data); records=discover_images(root,args.limit_per_folder); train_records,test_records=split_by_class(records,args.test_fraction)
    if not train_records: raise SystemExit(f"No training images found under {root}")
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model=ScoreUNet(args.base_channels).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    ema=copy.deepcopy(model).eval(); ema.requires_grad_(False); rng=random.Random(args.seed); model.train(); history=[]
    for epoch in range(args.epochs):
      losses=[]; rng.shuffle(train_records)
      for index,(path,label) in enumerate(train_records):
        clean=augment(load_image(path,args.image_size).to(device),rng)
        y=patchwise_forward(clean,1.0,args.patch,args.stride,seed=args.seed+epoch*len(train_records)+index)
        d=descriptor_maps(y,args.patch,args.stride).unsqueeze(0)
        t=float(rng.uniform(0.08,1.0))
        # A paired interpolation path ends at the exact degraded measurement used for conditioning.
        xt=(1.0-t)*clean+t*y
        pred=model(xt.unsqueeze(0),y.unsqueeze(0),d,torch.tensor([t],device=device))
        l1=F.l1_loss(pred,clean.unsqueeze(0)); mse=F.mse_loss(pred,clean.unsqueeze(0))
        loss=0.75*l1+0.20*mse+0.05*gradient_loss(pred,clean.unsqueeze(0))
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        with torch.no_grad():
          for ep,mp in zip(ema.parameters(),model.parameters()): ep.mul_(args.ema_decay).add_(mp,alpha=1-args.ema_decay)
        losses.append(float(loss.detach()))
      mean_loss=float(np.mean(losses)); history.append(mean_loss)
      print(f"epoch {epoch+1}/{args.epochs} loss={mean_loss:.6f}")
    out=Path(args.outputs); (out/'training').mkdir(parents=True,exist_ok=True); (out/'models').mkdir(parents=True,exist_ok=True)
    checkpoint={"model":ema.state_dict(),"config":vars(args),"train_count":len(train_records),"test_count":len(test_records)}
    torch.save(checkpoint,out/'models'/'lafd_score_unet.pt')
    with (out/'training'/'history.csv').open('w',newline='') as f:
      w=csv.writer(f); w.writerow(['epoch','loss']); w.writerows(enumerate(history,1))
    (out/'training'/'split.json').write_text(json.dumps({"train":[str(p) for p,_ in train_records],"test":[str(p) for p,_ in test_records]},indent=2))
    print(f"Saved EMA checkpoint to {out/'models'/'lafd_score_unet.pt'}")

def restore(model,measurement,descriptors,steps,seed,stochastic):
    device=measurement.device; x=measurement.clone(); gen=torch.Generator(device=device); gen.manual_seed(seed)
    # Euler integration along the paired clean-to-measurement interpolation path.
    with torch.no_grad():
      for i in range(steps):
        t=1.0-i/steps
        if stochastic and i>0: x=x+torch.randn(x.shape,device=device,generator=gen)*0.0015
        clean_hat=model(x.unsqueeze(0),measurement.unsqueeze(0),descriptors.unsqueeze(0),torch.tensor([t],device=device))[0]
        x=x+(clean_hat-measurement)/steps
    return x.clamp(0,1)

def evaluate(args):
    out=Path(args.outputs); ckpt=torch.load(out/'models'/'lafd_score_unet.pt',map_location='cpu',weights_only=False)
    conf=ckpt['config']; device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model=ScoreUNet(conf['base_channels']).to(device); model.load_state_dict(ckpt['model']); model.eval()
    _,tests=split_by_class(discover_images(Path(args.data),args.limit_per_folder),args.test_fraction)
    from .metrics import image_metrics,frequency_error
    from .diffusion import scalar_denoise,global_wiener_restore
    rows=[]
    for idx,(path,label) in enumerate(tests):
      clean=load_image(path,conf['image_size']).to(device)
      y=patchwise_forward(clean,1.0,conf['patch'],conf['stride'],seed=args.seed+idx)
      desc=descriptor_maps(y,conf['patch'],conf['stride']); runs=[]
      for run in range(args.runs): runs.append(restore(model,y,desc,args.steps,args.seed+idx*100+run,True))
      stack=torch.stack(runs); estimate=stack.mean(0).cpu().numpy(); variance_map=stack.var(0).mean(0).cpu().numpy()
      ref=clean.cpu().numpy(); degraded=y.cpu().numpy()
      scalar=scalar_denoise(y).cpu().numpy(); global_fdm=global_wiener_restore(y).cpu().numpy()
      name=f"{idx+1:02d}_{label}_{path.stem}"
      save_gray(out/'restorations'/f'{name}_clean.png',ref); save_gray(out/'restorations'/f'{name}_measurement.png',degraded)
      save_gray(out/'restorations'/f'{name}_lafd.png',estimate); save_gray(out/'restorations'/f'{name}_uncertainty.png',variance_map/(variance_map.max()+1e-8))
      for method,img in [('scalar_proxy',scalar),('global_fdm_proxy',global_fdm),('lafd',estimate)]:
        m=image_metrics(ref,img,variance_map if method=='lafd' else None); m.update(method=method,image=name,**{'class':label})
        m['local_frequency_error']=frequency_error(ref,img,conf['patch'],conf['stride']); rows.append(m)
    metric_path=out/'metrics'/'evaluation.csv'; metric_path.parent.mkdir(parents=True,exist_ok=True)
    with metric_path.open('w',newline='') as f:
      fields=['image','class','method','mse','bias','variance','rmse','psnr','ssim','local_frequency_error']
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Evaluated {len(tests)} test images; metrics: {metric_path}")

def main():
    p=argparse.ArgumentParser(description='Locally Adaptive Fourier Diffusion restoration'); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--data',default='lung_colon_image_set'); t.add_argument('--outputs',default='outputs'); t.add_argument('--limit-per-folder',type=int,default=5); t.add_argument('--test-fraction',type=float,default=.2); t.add_argument('--image-size',type=int,default=256); t.add_argument('--patch',type=int,default=64); t.add_argument('--stride',type=int,default=32); t.add_argument('--epochs',type=int,default=40); t.add_argument('--lr',type=float,default=3e-4); t.add_argument('--base-channels',type=int,default=16); t.add_argument('--ema-decay',type=float,default=.99); t.add_argument('--seed',type=int,default=7); t.add_argument('--device',default=None); t.set_defaults(func=train)
    e=sub.add_parser('evaluate'); e.add_argument('--data',default='lung_colon_image_set'); e.add_argument('--outputs',default='outputs'); e.add_argument('--limit-per-folder',type=int,default=5); e.add_argument('--test-fraction',type=float,default=.2); e.add_argument('--steps',type=int,default=8); e.add_argument('--runs',type=int,default=4); e.add_argument('--seed',type=int,default=100); e.add_argument('--device',default=None); e.set_defaults(func=evaluate)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
