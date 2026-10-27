import numpy as np
from skimage.metrics import structural_similarity

def image_metrics(reference, estimate, sample_variance=None):
    ref=np.asarray(reference,dtype=np.float32); est=np.asarray(estimate,dtype=np.float32)
    err=est-ref; bias=float(np.mean(err**2))
    variance=float(np.mean(sample_variance)) if sample_variance is not None else 0.0
    mse=bias+variance
    psnr=float(10*np.log10(1.0/max(mse,1e-12)))
    ssim=float(structural_similarity(ref,est,data_range=1.0,channel_axis=0 if ref.ndim==3 else None))
    return {"mse":mse,"bias":bias,"variance":variance,"rmse":float(np.sqrt(mse)),"psnr":psnr,"ssim":ssim}

def frequency_error(reference, estimate, patch=64, stride=32):
    """Mean relative patch-spectrum discrepancy; a lightweight proxy, not calibrated MTF/NPS."""
    import torch
    from .diffusion import patch_starts
    a=torch.as_tensor(reference,dtype=torch.float32); b=torch.as_tensor(estimate,dtype=torch.float32)
    if a.ndim==2: a=a[None]; b=b[None]
    vals=[]
    for y in patch_starts(a.shape[-2],patch,stride):
      for x in patch_starts(a.shape[-1],patch,stride):
        pa=a[:,y:y+patch,x:x+patch]; pb=b[:,y:y+patch,x:x+patch]
        fa=torch.fft.rfft2(pa).abs().mean(0); fb=torch.fft.rfft2(pb).abs().mean(0)
        vals.append(float(((fa-fb).abs()/(fa.mean()+1e-6)).mean()))
    return float(np.mean(vals)) if vals else 0.0
