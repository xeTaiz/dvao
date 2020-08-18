import torch, torch.nn, torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from pathlib   import Path
from argparse  import ArgumentParser, Namespace
from itertools import count
import time, os

from train import Unet3D, QureDataset

def predict(args):
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(args.checkpoint)
    if args.out is None: out_path = checkpoint_path.parent/checkpoint_path.stem
    else:                out_path = Path(args.out)
    if not out_path.exists(): out_path.mkdir()
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # model = Unet3D.load_from_checkpoint(checkpoint_path=args.checkpoint)
    model = Unet3D(Namespace(**ckpt['hparams']))
    model.load_state_dict(ckpt['state_dict'])

    model.to(dev)
    model.eval()
    model.freeze()

    item_path = Path(args.item_path)
    if args.valid:
        items = model.items[model.split_idx:]
        ds_path = args.ds_path
    elif item_path.is_dir():
        items = [n for n in os.listdir(item_path)]
        ds_path = item_path
    else:
        items = [item_path]
        ds_path = item_path.parent
    ds = QureDataset(ds_path, items=items, output_meta=True,
            tf_as_pts=model.tf_as_pts, vol_sz=model.hparams.vol_sz,
            device=torch.device('cpu') if args.tfm_cpu else dev)

    for i, batch in tqdm(enumerate(ds)):
        vol, ao, tf, meta = batch
        if not model.tf_as_pts: tf = tf[None]
        elif   model.tf_as_pts: tf = [tf.to(dev)]
        if i > args.only and args.only > 0: break
        tf_name  = meta['ao_uuid']
        out_name = f"Prediction_{meta['name']}_{tf_name}.pt"
        vol_in = vol.squeeze()[None, None].to(dev).float()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record() # Log Time
        pred = model.forward(vol_in, tf)                  # Model.forward()
        end.record()
        torch.cuda.synchronize()
        dur = start.elapsed_time(end)

        torch.save({
            'pred': pred.to(torch.float16).cpu(),
            'vol':  vol.to(torch.float16).cpu(),
            'gt':   ao.to(torch.float16).cpu(),
            'tf':   tf[0].cpu() if isinstance(tf, list) else tf.cpu(),
            **meta
        }, out_path/out_name)
        tqdm.write(f"Saved prediction for {meta['name']} (TF={tf_name}) as {out_name}. Inferred in {dur}ms.")

if __name__ == '__main__':
    parser = ArgumentParser('Infer DVAO')
    parser.add_argument('checkpoint',   type=str, help='Path to model checkpoint')
    parser.add_argument('item_path',      type=str, help='Path to Input Item')
    parser.add_argument('--out',        type=str, default=None, help='Path where the output predictions are saved to')
    parser.add_argument('--only',       type=int, default=0, help='Number of volumes to predict from the ds')
    parser.add_argument('--valid', action='store_true', help='Whether to use the validation items according to the training runs split')
    parser.add_argument('--tfm_cpu', action='store_true', help='Whether the data preprocessing (cropping, resizing, ..) is done on CPU (to save GPU memory)')
    args = parser.parse_args()

    predict(args)
