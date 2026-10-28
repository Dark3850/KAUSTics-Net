# Fixed experiment data

Split manifests are tracked in Git; image pixels are hosted separately on Hugging Face. Use `scripts/download_assets.py` after publishing, or point training directly at the prepared payload:

```bash
python train.py --manifest huggingface/datasets/KAUSTics/manifest_256.json --seed 1 --output runs/kaustics_seed1
python finetune.py --manifest huggingface/datasets/Sea-Undistort-evaluation/manifest.json \
  --init checkpoints/kaustics/seed1.pt --seed 1 --output runs/sea_seed1
```

In the complete local bundle, pixels are stored only in the Hugging Face payload folders to avoid duplicating a full download. Use the manifest paths above before publication. The main README uses `data/` paths populated by the downloader after publication. All variants of a KAUSTics source image remain in one split; Sea-Undistort target paths always refer to original GT.
