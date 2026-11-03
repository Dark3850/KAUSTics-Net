# Selected result weights

Only six checkpoints are included in the complete local bundle, under `../huggingface/models/KAUSTics-Net/`: `kaustics/seed{1,2,3}.pt` and `sea_undistort/seed{1,2,3}.pt`. Upload that folder to Hugging Face, not the weights to GitHub. `selected.json` records epochs, checksums and tensor identities. This runtime directory is populated by the downloader after publication.

After downloading the lightweight GitHub ZIP, populate this directory with `python scripts/download_assets.py --namespace dar96io --kind checkpoints`.
