"""Download published assets into the portable runtime layout (no environment creation)."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--namespace',required=True,help='HF account or organization (replace the placeholder)')
    p.add_argument('--kind',choices=('kaustics','sea','checkpoints','all'),default='all')
    p.add_argument('--revision',default='main',help='Pin to a repo revision/commit when reproducing a release')
    p.add_argument('--original',action='store_true',help='Also download original-resolution KAUSTics images')
    args=p.parse_args()
    if args.namespace=='YOUR_HF_USERNAME':
        p.error('The repositories are not published by this package. Enter the actual publisher namespace.')
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        p.error('Install the optional downloader first: python -m pip install -U huggingface_hub')
    root=Path(__file__).resolve().parents[1]
    if args.kind in ('kaustics','all'):
        patterns=['rgb256/**','manifest_256.json','source_assignments.csv']
        if args.original:
            patterns+=['original/**','manifest_original.json']
        snapshot_download(repo_id=args.namespace+'/KAUSTics',repo_type='dataset',revision=args.revision,
                          local_dir=str(root/'data/kaustics'),allow_patterns=patterns)
    if args.kind in ('sea','all'):
        snapshot_download(repo_id=args.namespace+'/Sea-Undistort-evaluation',repo_type='dataset',revision=args.revision,
                          local_dir=str(root/'data/sea_undistort'),allow_patterns=['images/**','manifest.json'])
    if args.kind in ('checkpoints','all'):
        snapshot_download(repo_id=args.namespace+'/KAUSTics-Net',revision=args.revision,
                          local_dir=str(root/'checkpoints'),allow_patterns=['kaustics/*.pt','sea_undistort/*.pt','selected.json'])


if __name__=='__main__':
    main()
