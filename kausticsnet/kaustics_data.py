import random
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from .partitions import load_manifest, resolve_records
from .utils import key_seed

class Pairs(torch.utils.data.Dataset):
    def __init__(self, manifest_path, split, image_size, seed=1, prob_aug=0.6):
        path = Path(manifest_path).resolve()
        self.manifest = load_manifest(path)
        if self.manifest['image_size'] not in (None, image_size):
            raise ValueError('Manifest and requested resolution disagree')
        self.rows = []
        self.split, self.image_size, self.seed = split, image_size, seed
        self.prob_aug, self.epoch = prob_aug, 0
        for rec in sorted(resolve_records(self.manifest, path.parent, split), key=lambda r: r['patch_id']):
            for shape in (1, 2):
                variants = [v for v in rec['variants'] if v['shape'] == shape]
                base = next(v for v in variants if v['kind'] == 'base')
                choices = [base] if split == 'train' else variants
                for v in choices:
                    self.rows.append({'gt': rec['gt'], 'source_id': rec['source_id'],
                        'patch_id': rec['patch_id'], 'shape': shape, 'variant': v,
                        'augs': [a for a in variants if a['kind'] != 'base']})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        v = row['variant']
        if self.split == 'train':
            rng = random.Random(key_seed(self.seed, self.epoch, index, 'augmentation'))
            if rng.random() < self.prob_aug:
                v = rng.choice(row['augs'])
        def read(path):
            with Image.open(path) as im:
                im = im.convert('RGB')
                if im.size != (self.image_size, self.image_size):
                    im = im.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                return TF.to_tensor(im) * 2 - 1
        name = f"{row['patch_id']}_shape{row['shape']}_{v['kind']}"
        return read(v['path']), read(row['gt']), index, name, row['source_id']

