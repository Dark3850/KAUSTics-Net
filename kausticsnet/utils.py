import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

# Tracking is always disabled; this package does not import the W&B SDK.
os.environ.update(WANDB_DISABLED='true', WANDB_MODE='disabled', WANDB_SILENT='true')


def key_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2**31)


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_weights(path, model):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('prediction_type') != 'v_prediction' or saved.get('image_size') != 256:
        raise ValueError('Expected a released 256px velocity checkpoint')
    model.load_state_dict(saved['model'], strict=True)
    return saved


def rng_state(generator=None):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader=generator.get_state() if generator is not None else None)


def restore_rng(state, generator=None):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    if generator is not None and state['loader'] is not None:
        generator.set_state(state['loader'])


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
