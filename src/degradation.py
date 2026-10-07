import numpy as np
from scipy.ndimage import gaussian_filter

def fft_gaussian_kernel(shape, sigma):
    h,w=shape
    yy,xx=np.meshgrid(np.arange(h)-h//2,np.arange(w)-w//2,indexing="ij")
    k=np.exp(-(xx**2+yy**2)/(2*sigma**2))
    k/=k.sum()
    return np.fft.fft2(np.fft.ifftshift(k))

def local_texture_score(patch):
    gx=np.diff(patch,axis=1)
    gy=np.diff(patch,axis=0)
    grad=np.concatenate([gx.ravel(),gy.ravel()])
    return float(np.std(grad))

def estimate_local_profile(patch, strength=0.65, noise_sigma=0.08):
    t=np.clip(local_texture_score(patch)/0.25,0,1)
    blur_sigma=(1-t)*1.8+t*0.7
    noise_level=(1-t*0.55)*noise_sigma
    return blur_sigma, noise_level

def degrade_patch_fft(patch, sigma, noise_level, rng):
    h=fft_gaussian_kernel(patch.shape, sigma)
    x=np.fft.fft2(patch)
    blurred=np.real(np.fft.ifft2(x*h))
    white=rng.normal(0,1,patch.shape).astype(np.float32)
    nfreq=np.fft.fft2(white)
    freq_weight=np.abs(h)
    colored=np.real(np.fft.ifft2(nfreq*np.maximum(freq_weight,1e-3)))
    colored/=max(float(colored.std()),1e-6)
    y=np.clip(blurred+noise_level*colored,0,1)
    mtf=np.abs(h)
    nps=np.abs(np.fft.fft2(colored))**2
    nps=nps/(nps.mean()+1e-8)
    return y.astype(np.float32), mtf.astype(np.float32), nps.astype(np.float32)

def hann2d(n):
    a=np.hanning(n).astype(np.float32)
    w=np.outer(a,a)
    return np.maximum(w,1e-4)

def patchwise_degrade(image, patch_size=64, stride=32, strength=0.65, noise_sigma=0.08, seed=42):
    rng=np.random.default_rng(seed)
    h,w=image.shape
    out=np.zeros_like(image,dtype=np.float32)
    weight=np.zeros_like(image,dtype=np.float32)
    profiles=[]
    win=hann2d(patch_size)
    for r in range(0,h-patch_size+1,stride):
        for c in range(0,w-patch_size+1,stride):
            patch=image[r:r+patch_size,c:c+patch_size]
            sigma,nlvl=estimate_local_profile(patch,strength,noise_sigma)
            yp,mtf,nps=degrade_patch_fft(patch,sigma,nlvl,rng)
            out[r:r+patch_size,c:c+patch_size]+=yp*win
            weight[r:r+patch_size,c:c+patch_size]+=win
            profiles.append({"row":r,"col":c,"sigma":sigma,"noise":nlvl,"mtf":mtf,"nps":nps})
    out/=np.maximum(weight,1e-6)
    return np.clip(out,0,1).astype(np.float32), profiles

def scalar_degrade(image, sigma=1.5, noise_sigma=0.08, seed=42):
    rng=np.random.default_rng(seed)
    h=fft_gaussian_kernel(image.shape,sigma)
    y=np.real(np.fft.ifft2(np.fft.fft2(image)*h))
    noise=rng.normal(0,noise_sigma,image.shape)
    return np.clip(y+noise,0,1).astype(np.float32)

def global_fft_degrade(image, sigma=1.2, noise_sigma=0.08, seed=42):
    rng=np.random.default_rng(seed)
    h=fft_gaussian_kernel(image.shape,sigma)
    white=rng.normal(0,1,image.shape)
    colored=np.real(np.fft.ifft2(np.fft.fft2(white)*np.abs(h)))
    colored/=max(colored.std(),1e-6)
    y=np.real(np.fft.ifft2(np.fft.fft2(image)*h))+noise_sigma*colored
    return np.clip(y,0,1).astype(np.float32)
