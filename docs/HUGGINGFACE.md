# Publishing this release

The upload folders are prepared locally. Nothing has been uploaded to GitHub or Hugging Face by this packaging task. Replace `YOUR_HF_USERNAME` with your account/organization in this guide and the main README.

## What goes where

| Destination | Local content | Approximate logical size |
|---|---|---:|
| GitHub | code, requirements, README, figures, split manifests, documentation | see release report |
| HF model repo `YOUR_HF_USERNAME/KAUSTics-Net` | `huggingface/models/KAUSTics-Net/` |151.4MiB weights |
| HF dataset repo `YOUR_HF_USERNAME/KAUSTics` | `huggingface/datasets/KAUSTics/` |24.54GiB images |
| HF dataset repo `YOUR_HF_USERNAME/Sea-Undistort-evaluation` | `huggingface/datasets/Sea-Undistort-evaluation/` |1.15GiB images |

The KAUSTics repo includes both original patch bytes (~22.74GiB) and the exact256 cache (~1.79GiB). Readers can download only256 with `scripts/download_assets.py`; `--original` additionally fetches the original patches. Sea-Undistort images preserve their input bytes and are resized on loading. Its upstream dataset is also already hosted at https://huggingface.co/datasets/maxkromer/Sea-Undistort . Our payload is the840/240/120 split plus extra test input variants needed to reproduce this study, not a new dataset claim.

## Is it free?

As checked on 28 October 2026, Hugging Face offers free **best-effort public storage** and 100 GB private storage for free users/organizations. Public hosting is not an unlimited guaranteed allocation: the policy asks for responsible community-useful uploads beyond the first few GB and may require upgrading for substantial storage. Our complete dataset payload is about 25.7 GiB; do not assume this whole amount is guaranteed free. See the current [storage policy](https://huggingface.co/docs/hub/storage-limits).

To minimize the initial public payload, upload only the256 cache and the six model weights first (~1.94GiB combined), then add original patches if your account’s storage plan permits. This does not change the256 experiment data. Training/inference GPU hosting is a separate service; storing files does not run your model.

## Upload and get shareable links

1. Create a free account at https://huggingface.co/join . Create a write/fine-grained token for your own repositories at https://huggingface.co/settings/tokens . Keep it out of code and Git.
2. Use the current HF CLI (no new conda environment is needed). In a terminal, install `huggingface_hub` and log in interactively:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

3. From the **complete Ibex `github` folder**, upload each prepared folder to its own repository. The small GitHub code ZIP deliberately excludes these bulk payloads:

```bash
hf upload YOUR_HF_USERNAME/KAUSTics-Net huggingface/models/KAUSTics-Net .
hf upload YOUR_HF_USERNAME/KAUSTics huggingface/datasets/KAUSTics . --repo-type dataset
hf upload YOUR_HF_USERNAME/Sea-Undistort-evaluation huggingface/datasets/Sea-Undistort-evaluation . --repo-type dataset
```

These commands can create repositories automatically. You can instead create a Model or Dataset repository in the website first and select public/private visibility. If interrupted, repeat the same upload; the current client resumes and skips already committed files. Current [upload documentation](https://huggingface.co/docs/huggingface_hub/en/guides/upload) recommends `hf upload`, which also handles large folders.

For a smaller first KAUSTics upload, omit the original images and original manifest:

```bash
hf upload YOUR_HF_USERNAME/KAUSTics huggingface/datasets/KAUSTics . --repo-type dataset \
  --exclude 'original/**' --exclude 'manifest_original.json'
```

The dataset card describes the complete prepared payload; if publishing only256, edit its "Availability" paragraph to state that original patches are not uploaded yet. Check the three HF README cards and choose licenses for your own code/data/weights before public publication. Sea-Undistort retains the upstream attribution and CC BY-NC-SA4.0 terms; it is not relicensed here.

4. Share these ordinary public page links (no recipient account/token required for public ungated repos):

```text
https://huggingface.co/YOUR_HF_USERNAME/KAUSTics-Net
https://huggingface.co/datasets/YOUR_HF_USERNAME/KAUSTics
https://huggingface.co/datasets/YOUR_HF_USERNAME/Sea-Undistort-evaluation
```

A direct file link has the form `https://huggingface.co/YOUR_HF_USERNAME/KAUSTics-Net/resolve/main/kaustics/seed1.pt`. For a publication, record/tag the uploaded revision and use its commit hash rather than a mutable `main` URL. Private repositories require access authorization; do not share your write token.

## Upload code to GitHub

Download `KAUSTics-Net-github.zip` (the small code archive next to this folder), extract it, and create an empty GitHub repository. After replacing the placeholder URLs and choosing the authors’ license, run from the extracted folder:

```bash
git init
git add .
git commit -m "Release KAUSTics-Net training and fine-tuning code"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/KAUSTics-Net.git
git push -u origin main
```

The supplied `.gitignore` excludes all dataset pixels, weights, upload payloads, training outputs and caches. Never force-add those directories. Competitor repositories are linked rather than copied.

## Local Ibex storage

The prepared payload uses **regular hardlinked image files**, not symlinks. Their bytes already exist in the experiment data; the release does not duplicate those image blocks. Treat these staged images as immutable: writing a hardlinked file in place also changes its original. Downloading/copying the folder produces ordinary independent files at the destination. No original files were changed or deleted. A full download is ~25.8GiB; the small GitHub ZIP excludes all bulk images and weights.
