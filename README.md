# Locally Adaptive Fourier Diffusion (LAFD)

A compact, reproducible image-restoration project based on the architecture and objectives in the accompanying project report. It extends the global, shift-invariant Fourier Diffusion Model (FDM) idea with local texture and edge estimates, overlapping patch degradation, smooth overlap-add, and uncertainty visualization.

> **Scope of the supplied data:** the repository contains lung and colon histopathology images, not CT scans. The pipeline preserves their RGB stain colors and resizes them to 256x256 for a controlled restoration experiment. Synthetic blur and correlated noise are proxies for MTF/NPS degradation; they are not a calibrated scanner model and the results are not for clinical use.

## Problem statement

Global Fourier degradation assigns one modulation transfer function (MTF) and noise power spectrum (NPS) to every pixel. Real imaging systems can vary across an image, while tissue boundaries and fine structures need different restoration behavior from smooth regions. The project implements a piecewise-stationary approximation: it measures local texture/edge activity, assigns each overlapping patch its own blur/noise levels, and blends patch results with a Hann window.

## How this extends the base paper

The report architecture diagram is included at [`outputs/figures/report_architecture.png`](outputs/figures/report_architecture.png). The base paper, M. Tivnan et al., “Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation,” *IEEE Transactions on Medical Imaging*, 44(9), 3694–3706, 2025, uses global LSI operators and stationary noise, enabling controlled frequency-domain degradation and measurement-initialized restoration. The report’s proposed enhancement (LAFD) makes these statistics spatially adaptive. This implementation adds:

- 64×64 overlapping patches (32-pixel stride by default) with content-dependent blur and noise from gradient variance and edge density (`src/lafd/diffusion.py`).
- Hann-window overlap-add so patch boundaries do not become hard seams (`patchwise_forward` in the same module).
- A time-conditioned U-Net that sees the current state, the measurement, and two local descriptor maps (`src/lafd/model.py`).
- Repeated stochastic restoration runs and a pixelwise variance heatmap (`evaluate` in `src/lafd/pipeline.py`).
- MSE, bias, variance, RMSE, PSNR, SSIM, and a local Fourier-spectrum discrepancy metric (`src/lafd/metrics.py`).

The implementation follows the report's four-module architecture. The four-resolution U-Net predicts the clean image from paired degraded measurements along a clean-to-measurement interpolation path. Training combines pixel and gradient losses and keeps an exponential-moving-average (EMA) checkpoint. Inference integrates the predicted clean direction with an Euler update. This is a practical paired-restoration implementation of the report design, not an exact reproduction of the FDM paper's SDE parameterization. `scalar_proxy` and `global_fdm_proxy` are measurement-only Gaussian-smoothing and global Wiener comparators, not independently trained diffusion baselines. The local frequency discrepancy is a spectrum proxy rather than calibrated MTF/NPS measurement. Dice is omitted because this dataset has no segmentation masks.

## Project structure

```text
src/
  run.py                  CLI entry point
  lafd/
    data.py               image discovery, five-per-folder cap, resize, split
    diffusion.py          local descriptors, FFT MTF/NPS proxy, patch blending
    model.py              sinusoidal time embedding and conditioned U-Net
    metrics.py            fidelity and local frequency metrics
    pipeline.py           training, reverse sampling, evaluation and file output
outputs/
  models/                 trained checkpoint (generated)
  training/               split and training metadata (generated)
  measurements/           reserved for exported paired data
  restorations/           clean, measurement, restoration, uncertainty PNGs
  metrics/                evaluation CSV
  figures/                reserved for plots
```

## Data handling

The default dataset path is `lung_colon_image_set/`. It has five class folders (`lung_aca`, `lung_n`, `lung_scc`, `colon_aca`, `colon_n`). At most **five images are selected from each leaf folder**, in sorted filename order, so the default run uses no more than 25 images total. Each RGB image is resized from 768x768 to 256x256 and normalized to [0, 1]. A deterministic per-class split reserves one image for testing when a class has more than one selected image. Images are loaded as needed; the complete dataset is not copied into the output tree.

## Modules and code map

1. **Data measurement and simulation** — `src/lafd/data.py` discovers images, enforces `--limit-per-folder`, preprocesses them, and makes class-stratified train/test splits. `src/lafd/diffusion.py` applies an FFT Gaussian MTF proxy and frequency-colored noise to make paired clean/degraded observations.
2. **Patchwise Fourier forward process** — `local_descriptors` estimates gradient variance and edge density. `patchwise_forward` maps those local values to per-patch blur/noise strengths, filters each patch in the Fourier domain, and combines them with Hann weights. `global_fourier_forward` supplies a globally stationary frequency comparator.
3. **Score model and training** — `src/lafd/model.py` defines a sinusoidal time encoder and U-Net. `descriptor_maps` and `train` in `src/lafd/pipeline.py` construct training inputs from the diffused state, degraded measurement, local descriptor maps, and diffusion time; Adam optimizes the denoising-direction prediction. The checkpoint is written to `outputs/models/lafd_score_unet.pt`.
4. **Reverse process, evaluation, and uncertainty** — `restore` iterates from the measurement through short reverse updates. `evaluate` repeats the stochastic sampler, averages its restorations, computes the pixelwise sample variance, saves images under `outputs/restorations/`, and writes `outputs/metrics/evaluation.csv`.

## Setup and run

Python 3.10+ is recommended. Install dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train (small defaults are intended for a quick demonstration):

```powershell
$env:PYTHONPATH = "src"
python src/run.py train --data lung_colon_image_set --limit-per-folder 5 --epochs 40
```

Restore held-out images and export evaluation results:

```powershell
python src/run.py evaluate --data lung_colon_image_set --limit-per-folder 5 --steps 8 --runs 4
```

Use `--device cuda` when a compatible GPU is available. `--patch`, `--stride`, `--image-size`, `--epochs`, and `--base-channels` adjust compute and model size. Evaluation uses the same deterministic per-class split and five-image-per-folder cap as training.

## Outputs

For each held-out test image, `outputs/restorations/` contains:

- `*_clean.png` — reference image after resizing and normalization.
- `*_measurement.png` — synthetic degraded observation.
- `*_lafd.png` — mean of the repeated LAFD restoration samples.
- `*_uncertainty.png` — normalized pixelwise sample variance (bright means more variation between runs).

`outputs/metrics/evaluation.csv` has one row per image and comparator, with MSE, bias, variance, RMSE, PSNR, SSIM, and local frequency error. Training metadata and the split are kept in `outputs/training/`. Model weights are separate in `outputs/models/`.

## References


- M. Tivnan et al., “Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation,” *IEEE Transactions on Medical Imaging*, 2025.
- J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” NeurIPS, 2020.
- Y. Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” 2020.
- Project design and requirements: `Report-1 (1) (2).docx` (external source document supplied by the project author).
### Generated demonstration examples

The checked output directory contains the current color-preserving demonstration run: 12 training epochs at 256x256, at most five selected inputs per class, eight reverse steps, and four stochastic samples per test image. It evaluates five held-out images, one from each class.

| Class | Reference | Simulated measurement | LAFD output | Uncertainty |
|---|---|---|---|---|
| colon_aca | [clean](outputs/restorations/01_colon_aca_colonca1_clean.png) | [measurement](outputs/restorations/01_colon_aca_colonca1_measurement.png) | [restoration](outputs/restorations/01_colon_aca_colonca1_lafd.png) | [variance map](outputs/restorations/01_colon_aca_colonca1_uncertainty.png) |
| colon_n | [clean](outputs/restorations/02_colon_n_colonn1_clean.png) | [measurement](outputs/restorations/02_colon_n_colonn1_measurement.png) | [restoration](outputs/restorations/02_colon_n_colonn1_lafd.png) | [variance map](outputs/restorations/02_colon_n_colonn1_uncertainty.png) |
| lung_aca | [clean](outputs/restorations/03_lung_aca_lungaca1_clean.png) | [measurement](outputs/restorations/03_lung_aca_lungaca1_measurement.png) | [restoration](outputs/restorations/03_lung_aca_lungaca1_lafd.png) | [variance map](outputs/restorations/03_lung_aca_lungaca1_uncertainty.png) |
| lung_n | [clean](outputs/restorations/04_lung_n_lungn1_clean.png) | [measurement](outputs/restorations/04_lung_n_lungn1_measurement.png) | [restoration](outputs/restorations/04_lung_n_lungn1_lafd.png) | [variance map](outputs/restorations/04_lung_n_lungn1_uncertainty.png) |
| lung_scc | [clean](outputs/restorations/05_lung_scc_lungscc1_clean.png) | [measurement](outputs/restorations/05_lung_scc_lungscc1_measurement.png) | [restoration](outputs/restorations/05_lung_scc_lungscc1_lafd.png) | [variance map](outputs/restorations/05_lung_scc_lungscc1_uncertainty.png) |

Full metrics are in [`outputs/metrics/evaluation.csv`](outputs/metrics/evaluation.csv). Across the five held-out RGB images, LAFD averaged 0.00104 MSE, 29.98 dB PSNR, and 0.835 SSIM. The scalar smoothing proxy averaged 0.00215 MSE, 27.48 dB PSNR, and 0.757 SSIM. These are preliminary results from a small split, not a broad performance claim.
