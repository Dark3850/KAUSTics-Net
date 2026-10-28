import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .kaustics_data import Pairs
from .utils import key_seed


def read_tensor(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
        array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).mul(2).sub(1)


class SeaPairs(torch.utils.data.Dataset):
    def __init__(self, manifest, split, variant='all_effects'):
        path = Path(manifest).resolve()
        self.root, self.split, self.epoch = path.parent, split, 0
        self.manifest = json.loads(path.read_text())
        if self.manifest['target'] != 'original_water_free_gt':
            raise ValueError('Sea-Undistort must use the original GT, never an input variant')
        self.variant = variant
        if split not in ('train','val','test') or (split != 'test' and variant != 'all_effects'):
            raise ValueError('Fine-tuning/selection use all-effects inputs; three variants are for test')
        membership = self.manifest['split_membership']
        sets = {k:set(v) for k,v in membership.items()}
        if any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Scene IDs overlap across partitions')
        if {k:len(v) for k,v in sets.items()} != dict(train=840,val=240,test=120):
            raise ValueError('Expected the fixed 840/240/120 scene split')
        self.rows = [r for r in self.manifest['rows'] if r['split'] == split]
        if len(self.rows) != len(sets[split]) or {r['name'] for r in self.rows} != sets[split]:
            raise ValueError('Duplicate or missing scene rows')
        for row in self.rows:
            if variant not in row['inputs'] or row['inputs'][variant]['path'] == row['gt']['path']:
                raise ValueError('Missing variant or input/GT path collision')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (read_tensor(self.root/row['inputs'][self.variant]['path']),
                read_tensor(self.root/row['gt']['path']), index, row['name'], int(Path(row['name']).stem))


def validation_indices(dataset):
    """The paper's fixed120 KAUSTics views; full240 SU scenes."""
    if isinstance(dataset, SeaPairs):
        return list(range(len(dataset)))
    groups = {}
    for index, row in enumerate(dataset.rows):
        key = (str(row['source_id']), int(row['shape']), row['variant']['kind'])
        groups.setdefault(key, []).append(index)
    if len(groups) != 30:
        raise ValueError('Expected 3 sources x 2 caustics x 5 variants')
    chosen = []
    for key in sorted(groups):
        def priority(i):
            row = dataset.rows[i]
            name = f"{row['patch_id']}_shape{row['shape']}_{row['variant']['kind']}"
            return key_seed('sampler-sweep-subset', name)
        chosen.extend(sorted(groups[key], key=priority)[:4])
    return sorted(chosen)
