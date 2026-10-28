"""Restore one image, an image directory, or a fixed dataset partition."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from kausticsnet.model import SlimUNet, DiffusionScheduler, set_seed, to_uint8_img
from kausticsnet.diffusion import pilot_ddim
from kausticsnet.data import Pairs, SeaPairs, read_tensor
from kausticsnet.engine import SAMPLER
from kausticsnet.utils import key_seed, load_weights, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path)
    source.add_argument('--manifest',type=Path)
    p.add_argument('--dataset',choices=('kaustics','sea_undistort'),default='kaustics')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--variant',choices=('all_effects','no_waves','no_sunglint'),default='all_effects')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--lpips',action='store_true',help='Include AlexNet LPIPS (first use downloads public weights)')
    p.add_argument('--limit',type=int,help='Optional smoke/example limit; omit for full evaluation')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output folder; predictions are never overwritten')
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device=torch.device(args.device)
    model=SlimUNet(image_size=256,time_emb_dim=256).to(device).eval()
    saved=load_weights(args.checkpoint,model)
    scheduler=DiffusionScheduler(1000); scheduler.to_device(device)
    perceptual=None
    if args.lpips:
        import lpips
        perceptual=lpips.LPIPS(net='alex').to(device).eval()
    if args.manifest:
        data = (SeaPairs(args.manifest,args.split,args.variant) if args.dataset=='sea_undistort'
                else Pairs(args.manifest,args.split,256))
        def examples():
            for i in range(len(data)):
                condition, target, _, name, source=data[i]
                yield condition,target,name,str(source)
    else:
        paths=sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
        paths=[x for x in paths if x.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff')]
        if not paths:
            p.error('No input images')
        def examples():
            for path in paths:
                yield read_tensor(path),None,path.name,''
    args.output.mkdir(parents=True)
    rows=[]
    with torch.no_grad():
        for index,(condition,target,name,source) in enumerate(examples()):
            if args.limit is not None and index>=args.limit:
                break
            set_seed(key_seed(saved['seed'],name,'inference'))
            result,_=pilot_ddim(model,scheduler,condition[None].to(device),**SAMPLER,
                                prediction_type='v_prediction',endpoint='legacy')
            rgb=to_uint8_img(result[0])
            Image.fromarray(rgb).save(args.output/(Path(name).stem+'.png'))
            if target is not None:
                gt=to_uint8_img(target)
                mse=float(np.square((rgb.astype(np.float64)-gt.astype(np.float64))/255).mean())
                row=dict(name=name,source_id=source,psnr=float(peak_signal_noise_ratio(gt,rgb,data_range=255)),
                    ssim=float(structural_similarity(gt,rgb,data_range=255,channel_axis=2)),mse=mse,rmse=mse**.5,
                    delta_e_2000=float(deltaE_ciede2000(rgb2lab(rgb/255.),rgb2lab(gt/255.)).mean()))
                if perceptual is not None:
                    pt=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device).float()/127.5-1
                    row['lpips']=float(perceptual(pt,target[None].to(device)).item())
                rows.append(row)
            print(index+1,name,flush=True)
    if rows:
        with (args.output/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_json(args.output/'summary.json',dict(count=len(rows),seed=saved['seed'],variant=args.variant,
            metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('name','source_id')}))


if __name__=='__main__':
    main()
