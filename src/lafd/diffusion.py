import math
import torch
import torch.nn.functional as F


def patch_starts(length, patch, stride):
    starts = list(range(0, max(length - patch, 0) + 1, stride))
    if not starts or starts[-1] != length - patch: starts.append(length - patch)
    return starts


def local_descriptors(image, patch=64, stride=32):
    """Overlapping [gradient variance, normalized edge density] descriptors and patch origins."""
    _, h, w = image.shape
    gx = F.pad(image[:, :, 1:] - image[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(image[:, 1:, :] - image[:, :-1, :], (0, 0, 0, 1))
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    result, positions = [], []
    for y in patch_starts(h, patch, stride):
        for x in patch_starts(w, patch, stride):
            v = mag[:, y:y+patch, x:x+patch]
            result.append(torch.stack((v.var(), (v > 0.08).float().mean())))
            positions.append((y, x))
    return torch.stack(result), positions


def _hann(patch, device):
    one = torch.hann_window(patch, periodic=False, device=device).clamp_min(1e-3)
    return (one[:, None] * one[None, :])[None, ...]


def _local_blur_noise(crop, blur_sigma, noise_sigma, generator=None):
    h, w = crop.shape[-2:]
    f = torch.fft.fftfreq(h, device=crop.device).reshape(-1, 1)
    g = torch.fft.rfftfreq(w, device=crop.device).reshape(1, -1)
    radius2 = f.square() + g.square()
    mtf = torch.exp(-2 * math.pi**2 * blur_sigma**2 * radius2)
    blurred = torch.fft.irfft2(torch.fft.rfft2(crop) * mtf, s=(h, w)).real
    white = torch.randn(crop.shape, device=crop.device, dtype=crop.dtype, generator=generator)
    # Smoothly colored NPS, stronger at low frequencies; normalized to unit spatial std.
    nps_filter = torch.exp(-radius2 / (2 * 0.12**2)).sqrt()
    noise = torch.fft.irfft2(torch.fft.rfft2(white) * nps_filter, s=(h, w)).real
    noise = noise / noise.std().clamp_min(1e-6)
    return blurred, noise * noise_sigma


def patchwise_forward(image, t, patch=64, stride=32, seed=None, return_descriptors=False):
    """LAFD forward state: local FFT blur/noise with smooth Hann overlap-add."""
    device = image.device
    descriptors, positions = local_descriptors(image, patch, stride)
    weights = torch.zeros_like(image); accum = torch.zeros_like(image)
    window = _hann(patch, device)
    generator = torch.Generator(device=device)
    if seed is not None: generator.manual_seed(int(seed))
    for desc, (y, x) in zip(descriptors, positions):
        texture, edges = desc
        # More edge/texture content receives less blur; flat areas receive stronger denoising.
        blur_sigma = (0.45 + 0.8 * (1.0 - edges.clamp(0, 1))).item() * float(t)
        noise_sigma = (0.006 + 0.016 * (1.0 - texture.clamp(0, 0.15) / 0.15)).item() * float(t)
        crop = image[:, y:y+patch, x:x+patch]
        blurred, noise = _local_blur_noise(crop, blur_sigma, noise_sigma, generator)
        wt = window
        accum[:, y:y+patch, x:x+patch] += (blurred + noise) * wt
        weights[:, y:y+patch, x:x+patch] += wt
    state = accum / weights.clamp_min(1e-8)
    if return_descriptors:
        return state, descriptors
    return state


def global_fourier_forward(image, t, noise=True):
    blur, nps = _local_blur_noise(image, 1.1 * float(t), 0.035 * float(t) if noise else 0.0)
    return blur + nps

def scalar_denoise(measurement):
    """Simple scalar-noise baseline: spatial Gaussian smoothing of the measurement."""
    import torch.nn.functional as F
    kernel=torch.tensor([1.,4.,6.,4.,1.],device=measurement.device,dtype=measurement.dtype)/16
    channels=measurement.shape[0]
    k=(kernel[:,None]*kernel[None,:])[None,None].expand(channels,1,5,5).contiguous()
    padded=F.pad(measurement[None],(2,2,2,2),mode='reflect')
    return F.conv2d(padded,k,groups=channels).squeeze(0)

def global_wiener_restore(measurement, blur_sigma=1.1, regularization=0.01):
    """Global stationary Fourier/Wiener proxy using one image-wide MTF and NPS prior."""
    import math
    h,w=measurement.shape[-2:]; device=measurement.device
    fy=torch.fft.fftfreq(h,device=device).reshape(-1,1); fx=torch.fft.rfftfreq(w,device=device).reshape(1,-1)
    r2=fy.square()+fx.square(); mtf=torch.exp(-2*math.pi**2*blur_sigma**2*r2)
    nps=torch.exp(-r2/(2*0.12**2))
    filt=mtf/(mtf.square()+regularization*nps+1e-8)
    restored=torch.fft.irfft2(torch.fft.rfft2(measurement)*filt,s=(h,w)).real
    return restored.clamp(0,1)
