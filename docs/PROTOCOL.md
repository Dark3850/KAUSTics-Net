# Released configuration

This package is extracted from the final revision code. Architecture, schedule, losses, DDIM spacing and differentiable fine-tuning sampler retain the original function bodies. The portable CLI replaces the dated cluster launch/audit chain. Historical run hashes and selected weights are recorded in `source_provenance.json` and `checkpoints/selected.json`.

## Shared model and inference

- RGB256 in [-1,1]; Pillow RGB conversion and bicubic resize when needed.
- Channels16/32/64/128/256, time embedding256, FiLM, learned conditional encoder and one-head cross-attention. No latent VAE or external diffusion-model weights.
- Cosine schedule,1,000 diffusion timesteps; velocity prediction with clean-image conversion `x0 = sqrt(alpha_bar)*xt - sqrt(1-alpha_bar)*v`.
- Locked sampler: strength0.95,25 rounded linearly spaced DDIM steps, guidance2.0. Its final destination uses alpha_bar[0], preserving the published endpoint convention.
- FP32; cudnn TF32 enabled, matrix-multiply TF32 disabled. Batch12; clip global gradient norm1; no AMP, EMA or LR decay. Same seeds1/2/3.
- Per-image inference noise derives from SHA256(training seed, image name, `inference`). Preserve filenames for matching replay. Runs on different hardware/software can differ numerically.

## KAUSTics training

1,000 epochs, fresh initialization; AdamW lr1e-4, weight decay1e-4; conditioning dropout0.1. Each epoch contains2,080 base pairs (1,040 references ×2 caustics). With probability0.6 the base input is replaced by one of its four augmentation variants; validation/test use all ten views per reference. All variants remain in the source image’s split.

Objective: 0.2 velocity MSE +0.8 reconstructed RGB MSE +0.2 low-frequency L1 +0.3(U L1+V L1) +0.2(1-SSIM) +0.2 normalized Lab L1. Gaussian low-frequency filter:21pixels/sigma5. RGB/low-frequency/UV reconstruction is unclipped; added SSIM/Lab use RGB clipped to[0,1].

Select maximum source-balanced sampled PSNR over the fixed120-view validation subset at epoch1/every25/final; earliest exact tie. Final validation/test have1,200 views each. The original1000px patch files are included for dataset completeness, but this is the **final256px model**, not the older native-resolution experiment.

## Sea-Undistort fine-tuning

Initialize from each seed’s corresponding KAUSTics checkpoint with a fresh AdamW optimizer, constant lr3e-5 and weight decay1e-4. Train on840 all-effects scenes, validate on240 all-effects scenes, evaluate120 test scenes under all-effects/no-waves/no-sunglint inputs. **All targets are the original GT**, never another distorted variant.

Keep the KAUSTics teacher-forced objective. Every fourth update (including step0), also backpropagate a sampled-output objective:0.8 RGB MSE +0.2(1-SSIM) +0.2 low-frequency normalized Lab L1, coefficient1. Sampling uses the fixed25-step sampler; only the final4 steps carry gradients. Sampled loss accumulates in microbatches4 before the shared clip/optimizer step. No paired spatial augmentation in this final configuration.

Validate epoch0,1 and every5 through100. Select maximum full240-scene validation SSIM; epoch0 eligible; earliest exact tie. The recorded rule allows stopping from epoch60 after five validation checks without an improvement of1e-4. All three released runs reached100. This is the fine-tuning rule, distinct from the competitors’ later patience10 protocol. The released final seeds were exploratory replications after prior test inspection; the benchmark was not blind.

| Released weight | Seed1 best epoch | Seed2 | Seed3 |
|---|---:|---:|---:|
| KAUSTics |975|925|925|
| Sea-Undistort fine-tuned |95|100|100|

## Running and storage

One A100 is the paper-matched setting; no multi-GPU launcher is required. Seeds are independent and can run on separate GPUs. CUDA installation is platform-dependent; e.g. install the pinned torch/torchvision wheels from the PyTorch cu126 index if appropriate for your drivers. The release was also checked on CPU; CPU is practical for smoke checks, not full training.

Each run keeps only `best.pt` and an overwritten `last.pt` containing optimizer and RNG state. `--resume` requires both. Release weights contain only model tensors and primitive metadata, load with `weights_only=True`, and omit training optimizer state. Their file hashes differ from the original archival checkpoints; tensor hashes are identical.

`infer.py` saves quantized RGB PNGs and, with paired manifests, PSNR/SSIM/normalized MSE/RMSE/DeltaE00, plus optional AlexNet LPIPS. Full non-reference metric machinery is intentionally outside the minimal train/fine-tune release. The paper’s broader evaluation stack is not claimed to be reproduced by this small script.
