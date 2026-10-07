# LAFD: Locally Adaptive Fourier Diffusion

Lightweight, laptop-safe implementation of the project described in the supplied Image Processing report.

## Scope
- Uses exactly 1,000 LC25000 images by default: 200 images per class.
- Resizes 768x768 JPEGs to 256x256.
- Simulates measurement degradation with local FFT-domain blur/noise.
- Uses overlapping 64x64 patches with Hann blending.
- Trains a small time-conditioned U-Net.
- Runs Euler–Maruyama-style reverse sampling.
- Produces restoration metrics, local MTF/NPS errors, uncertainty maps, and visual comparisons.
- Includes scalar and global-Fourier baselines as lightweight ablations.

## Classes
Lung benign tissue, lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, colon benign tissue.

## Important
This is an educational/research prototype. The LC25000 dataset contains histopathology images, while the report describes CT-style MTF/NPS motivation. The implementation therefore treats the LC25000 images as controlled image-restoration data rather than claiming clinical CT validity.
