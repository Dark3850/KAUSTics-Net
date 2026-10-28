"""Portable final-protocol runner. No cluster paths or experiment-manager imports."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .model import SlimUNet, DiffusionScheduler, set_seed, to_uint8_img
from .diffusion import pilot_loss_terms, pilot_objective, pilot_ddim
from .adaptation import sampled_restoration, sampled_loss
from .data import Pairs, SeaPairs, validation_indices
from .utils import (key_seed, cpu_state, load_weights, rng_state, restore_rng,
                    preserve_rng, atomic_save, write_json)

SAMPLER = dict(strength=.95, steps=25, guidance_scale=2.)


def teacher_backward(model, scheduler, batch, seed, epoch, step, device):
    condition, target = batch[0].to(device), batch[1].to(device)
    rng = torch.Generator(device=device).manual_seed(key_seed(seed, epoch, step, 0, 'noise'))
    condition = condition * (torch.rand((len(target),1,1,1),device=device,generator=rng) >= .1)
    t = torch.randint(1,1000,(len(target),),device=device,generator=rng)
    noise = torch.randn(target.shape, device=device, generator=rng)
    xt = scheduler.q_sample(target,t,noise)
    terms = pilot_loss_terms(scheduler,target,xt,model(xt,condition,t),noise,t,
                             prediction_type='v_prediction',rgb_loss='mse')
    loss = pilot_objective(terms)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite teacher loss')
    loss.backward()
    return float(loss.detach())


def update(model, optimizer, scheduler, batch, seed, epoch, step, device, finetune=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = teacher_backward(model,scheduler,batch,seed,epoch,step,device)
    if finetune and step % 4 == 0:
        model.eval()
        count = len(batch[0])
        with preserve_rng():
            for begin in range(0,count,4):
                set_seed(key_seed(seed,epoch,step,begin,'sampled-training-noise'))
                condition, target = batch[0][begin:begin+4].to(device),batch[1][begin:begin+4].to(device)
                raw = sampled_restoration(model,scheduler,condition)
                loss = sampled_loss(raw,target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite sampler-aware loss')
                fraction = len(condition)/count
                (loss*fraction).backward()
                total += float(loss.detach())*fraction
        model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def evaluate(model, data, indices, scheduler, seed, device):
    if data.split != 'val':
        raise ValueError('Training checkpoint selection is validation-only')
    model.eval()
    rows = []
    with preserve_rng():
        for i in indices:
            condition, target, _, name, source = data[i]
            set_seed(key_seed(seed,name,'inference'))
            predicted,_ = pilot_ddim(model,scheduler,condition[None].to(device),**SAMPLER,
                                     prediction_type='v_prediction',endpoint='legacy')
            pred, gt = to_uint8_img(predicted[0]), to_uint8_img(target)
            rows.append(dict(source=str(source), psnr=float(peak_signal_noise_ratio(gt,pred,data_range=255)),
                             ssim=float(structural_similarity(gt,pred,data_range=255,channel_axis=2))))
    # KAUSTics source-balanced selection; SU averages scenes.
    if isinstance(data,SeaPairs):
        return {k:sum(r[k] for r in rows)/len(rows) for k in ('psnr','ssim')}
    sources = sorted({r['source'] for r in rows})
    return {k:sum(sum(r[k] for r in rows if r['source']==s)/sum(r['source']==s for r in rows)
                  for s in sources)/len(sources) for k in ('psnr','ssim')}


def main(finetune=False):
    parser = argparse.ArgumentParser(description='Final KAUSTics-Net sampler-aware fine-tuning' if finetune else 'Final KAUSTics-Net training')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(1,2,3), required=True)
    parser.add_argument('--init', type=Path, required=finetune, help='Matching-seed KAUSTics checkpoint (fine-tuning only)')
    parser.add_argument('--epochs', type=int, default=100 if finetune else 1000)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true',help='Resume this output/last.pt; retains optimizer and RNG')
    parser.add_argument('--smoke',action='store_true',help='One optimizer update and one validation image; separate output required')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; CPU is supported for smoke checks with --device cpu')
    if args.epochs < 1 or (finetune and args.epochs > 100):
        parser.error('Use positive epochs; released fine-tuning protocol is capped at100')
    if args.output.exists() and not args.resume:
        parser.error('Output exists; choose a new folder or use --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(args.seed)
    model = SlimUNet(image_size=256,time_emb_dim=256).to(device)
    if finetune:
        state = load_weights(args.init,model)
        if state['seed'] != args.seed or state['family'] != 'kaustics':
            raise ValueError('Fine-tuning requires the matching-seed KAUSTics-trained checkpoint')
    scheduler = DiffusionScheduler(1000); scheduler.to_device(device)
    train = SeaPairs(args.manifest,'train') if finetune else Pairs(args.manifest,'train',256,args.seed,.6)
    val = SeaPairs(args.manifest,'val') if finetune else Pairs(args.manifest,'val',256)
    if len(train) != (840 if finetune else 2080) or len(val) != (240 if finetune else 1200):
        raise ValueError('Wrong training/validation inventory')
    sampler = DistributedSampler(train,num_replicas=1,rank=0,seed=args.seed,drop_last=False)
    generator = torch.Generator().manual_seed(key_seed(args.seed,0,'loader'))
    batch_size = 1 if args.smoke else 12
    loader = DataLoader(train,batch_size=batch_size,sampler=sampler,num_workers=args.workers,
                        pin_memory=device.type=='cuda',generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5 if finetune else 1e-4,weight_decay=1e-4)
    indices = validation_indices(val)
    if args.smoke:
        indices = indices[:1]
    score_key = 'ssim' if finetune else 'psnr'
    manifest_sha = __import__('hashlib').sha256(args.manifest.read_bytes()).hexdigest()
    config = dict(seed=args.seed,family='sea_undistort' if finetune else 'kaustics',
        image_size=256,prediction_type='v_prediction',sampler=SAMPLER,
        max_epochs=args.epochs,manifest_sha256=manifest_sha,smoke=args.smoke,
        selection=score_key,precision='fp32',torch=torch.__version__,batch_size=batch_size,
        lr=optimizer.param_groups[0]['lr'],weight_decay=1e-4,wandb='disabled')
    start, best, best_epoch, anchor, stale, stopped = 0,-float('inf'),0,-float('inf'),0,False
    history = []

    def save_best(epoch,metrics):
        atomic_save(dict(model=cpu_state(model),epoch=epoch,seed=args.seed,
                         family=config['family'],prediction_type='v_prediction',image_size=256,
                         sampler={**SAMPLER,'endpoint':'legacy'},validation=metrics),args.output/'best.pt')

    if args.resume:
        saved = torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        # last.pt is a trusted locally generated optimizer/RNG checkpoint, not a Hub download.
        if any(saved['config'][k] != config[k] for k in ('seed','family','manifest_sha256','smoke','max_epochs')):
            raise ValueError('Resume configuration differs')
        model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'],generator)
        start,best,best_epoch,anchor,stale,stopped = saved['control']
        history = saved['history']
        if not (args.output/'best.pt').exists():
            raise FileNotFoundError('Resume requires both best.pt and last.pt')
    elif finetune and not args.smoke:
        metrics = evaluate(model,val,indices,scheduler,args.seed,device)
        best = anchor = metrics[score_key]; save_best(0,metrics)
        history.append(dict(epoch=0,val_psnr=metrics['psnr'],val_ssim=metrics['ssim']))
    write_json(args.output/'config.json',config)
    end = 1 if args.smoke else args.epochs
    for epoch in range(start+1,end+1):
        if stopped:
            break
        train.epoch = epoch; sampler.set_epoch(epoch)
        total = count = 0
        for step,batch in enumerate(loader):
            value = update(model,optimizer,scheduler,batch,args.seed,epoch,step,device,finetune)
            total += value*len(batch[0]); count += len(batch[0])
            if args.smoke:
                break
        row = dict(epoch=epoch,train_loss=total/count,train_samples=count)
        validate = args.smoke or epoch==end or epoch==1 or epoch%(5 if finetune else 25)==0
        if validate:
            metrics = evaluate(model,val,indices,scheduler,args.seed,device)
            row.update(val_psnr=metrics['psnr'],val_ssim=metrics['ssim'])
            if metrics[score_key] > best:
                best,best_epoch = metrics[score_key],epoch
                save_best(epoch,metrics)
            if finetune:
                if metrics[score_key] >= anchor+1e-4:
                    anchor,stale = metrics[score_key],0
                else:
                    stale += 1
                stopped = epoch>=60 and stale>=5  # Original FT rule: five validation checks.
        history.append(row)
        atomic_save(dict(model=cpu_state(model),optimizer=optimizer.state_dict(),rng=rng_state(generator),
            config=config,control=(epoch,best,best_epoch,anchor,stale,stopped),history=history),args.output/'last.pt')
        columns = ['epoch','train_loss','train_samples','val_psnr','val_ssim']
        with (args.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(history)
        print(json.dumps(dict(**row,best_epoch=best_epoch,best=best)),flush=True)
    write_json(args.output/'completed.json',dict(best_epoch=best_epoch,best=best,early_stopped=stopped,
                                                smoke=args.smoke,epochs=history[-1]['epoch']))
