import numpy as np
from skimage.metrics import structural_similarity

def mse(a,b): return float(np.mean((a-b)**2))
def rmse(a,b): return float(np.sqrt(mse(a,b)))
def psnr(a,b):
    m=mse(a,b)
    return float(10*np.log10(1.0/max(m,1e-12)))
def ssim(a,b): return float(structural_similarity(a,b,data_range=1.0))
def bias_variance(samples,gt):
    mean=samples.mean(0)
    bias=float(np.mean((mean-gt)**2))
    var=float(np.mean((samples-mean[None])**2))
    return bias,var

def radial_profile(arr):
    h,w=arr.shape
    y,x=np.indices((h,w)); cy,cx=h//2,w//2
    r=np.sqrt((x-cx)**2+(y-cy)**2).astype(int)
    t=np.bincount(r.ravel(),arr.ravel())
    n=np.bincount(r.ravel())
    return t/np.maximum(n,1)

def local_mtf_nps_error(restored, target, patch=64, stride=32):
    vals=[]
    mtf_err=[]; nps_err=[]
    for r in range(0,restored.shape[0]-patch+1,stride):
        for c in range(0,restored.shape[1]-patch+1,stride):
            a=restored[r:r+patch,c:c+patch]
            b=target[r:r+patch,c:c+patch]
            fa=np.abs(np.fft.fft2(a)); fb=np.abs(np.fft.fft2(b))
            mtf_a=radial_profile(fa/(fa.max()+1e-8)); mtf_b=radial_profile(fb/(fb.max()+1e-8))
            nps_a=radial_profile(np.abs(np.fft.fft2(a-a.mean()))**2)
            nps_b=radial_profile(np.abs(np.fft.fft2(b-b.mean()))**2)
            mtf_err.append(float(np.mean((mtf_a-mtf_b)**2)))
            nps_err.append(float(np.mean((nps_a/(nps_a.mean()+1e-8)-nps_b/(nps_b.mean()+1e-8))**2)))
    return float(np.mean(mtf_err)), float(np.mean(nps_err))

def dice(a,b,thr=0.5):
    aa=a>thr; bb=b>thr
    return float(2*np.logical_and(aa,bb).sum()/max(aa.sum()+bb.sum(),1))
