#%% Imports
# Torch Stuff
import torch
import torch.nn.functional as F
from torch    import nn
from torch.nn import Linear, Conv1d, BatchNorm1d, Conv3d, InstanceNorm3d, AdaptiveAvgPool1d, ModuleList
from torch.utils.data import Dataset, DataLoader
# Third party torch stuff
import pytorch_lightning as pl
from pytorch_lightning.logging import CometLogger
from pytorch_msssim import ssim as pt_ssim

import numpy as np
# Python standard libraries
import math, os, sys, random
from pathlib      import Path
from collections  import OrderedDict
from argparse     import ArgumentParser
from functools    import partial, reduce
from time         import time
from enum         import Enum
# Local
from ranger        import Ranger
from torchinterp1d import Interp1d
from tf_utils      import read_inviwo_tf, make_tf_tex_from_pts_torch, apply_tf_torch, compute_normals
from ssim3d_torch  import ssim3d

#%% Data
def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def make_5D(t):
    if t.ndim < 5: return make_5D(t[None])
    else:          return t

def make_ND(t, N):
    if t.ndim < N: return make_ND(t[None])
    else:          return t

def get_crop(t, min_i, max_i):
    ''' Crops `t` in the last 3 dimensions for given 3D `min_i` and `max_i` like t[..., min_i[j]:max_i[j],..]'''
    return t[..., min_i[0]:max_i[0], min_i[1]:max_i[1], min_i[2]:max_i[2]]

class QureDataset(Dataset):     # Can load the data from DVS_RandomTF with all Random trapeze-like TFs
    def __init__(self, path, items=None, crop=True, vol_sz=64, tf_as_pts=True,
                output_meta=False, replace_max_int=True, device=torch.device('cpu')):
        super().__init__()
        self.path = Path(path)
        self.crop = crop
        self.dev = device
        self.replace_max_int = replace_max_int
        self.output_meta = output_meta
        self.vol_sz = vol_sz
        self.tf_as_pts = tf_as_pts
        if items is None:
            self.items = [n for n in os.listdir(path) if n.endswith('_original.pt')]
        else:
            self.items = items
        self.label = OrderedDict({
            orig: [n for n in os.listdir(path) if n.startswith(orig[:-12]) and not n.endswith('_original.pt')]
            for orig in self.items
        })
        for n, labs in self.label.items():
            if len(labs) == 0: self.items.remove(n)
        self.label_list = list(self.label.items())
        self.orig_len  = len(self.items)
        self.label_len = sum([len(v) for v in self.label.values()])

    def get_center_crop(self, t):
        return self.get_crop_around(t, (torch.Tensor([*t.shape]) // 2).long())

    def get_crop_around(self, t, mid):
        return t[mid[0] - self.vol_sz//2  :  mid[0] + self.vol_sz//2,
                 mid[1] - self.vol_sz//2  :  mid[1] + self.vol_sz//2,
                 mid[2] - self.vol_sz//2  :  mid[2] + self.vol_sz//2]

    def get_crop_resize(self, t, *additional):
        ''' Crops input tensor `t` so that all padding zeros are removed, then rescale to vol_sz³
        Args:
            t (Tensor): Input tensor to be cropped and resized to vol_sz
            additional (List of Tensors): Additional tensors that are cropped and resized like `t`.
        Returns: The cropped and resized tensor `t` and all the cropped/resized `additional`s
        '''
        nz = t.squeeze().nonzero() # Crop out axis aligned box which is tightest fit around nonzero intensities
        min_i, max_i = nz.min(dim=0).values, nz.max(dim=0).values + 1
        # Index last 3 dimensions according to where nonzero values are
        target_sz = (self.vol_sz, self.vol_sz, self.vol_sz)
        crop_fn   = partial(get_crop, min_i=min_i, max_i=max_i)
        resize_fn = partial(F.interpolate, size=target_sz, align_corners=True, mode='trilinear')
        tfms = compose(resize_fn, make_5D, crop_fn)
        return tuple(map(tfms, (t, ) + additional))

    def __getitem__(self, i):
        # Get indices and names
        orig_name = self.items[i]
        tf_idx  = torch.randint(len(self.label[orig_name]), (1,)).item()
        # Load from disk, unpack dict
        x     = torch.load(self.path/orig_name)
        label = torch.load(self.path/self.label[orig_name][tf_idx])
        ao    = label['ao'].to(self.dev)

        if self.replace_max_int: x[x.float()==1.0] = 0.0
        tf = torch.cat([torch.Tensor([[0.0, 0.0]]), label['tf_pts'][:, [0,-1]].float(), torch.Tensor([[1.0, 0.0]])])
        if self.tf_as_pts: tf = tf.to(self.dev)
        else:              tf = label['tf_tex'][None].to(self.dev) # Make texture
        # Transform, Crop, Cast and return
        x = x.to(self.dev)
        if x.device == torch.device('cpu'):
            xt, at = x.dtype, ao.dtype
            x = x.float()
            ao = ao.float()

        if   self.crop:
            out_x, out_y = self.get_crop_resize(x, ao)
        else:
            out_x, out_y = make_5D(x), make_5D(ao)
        if x.device == torch.device('cpu'):
            out_x = out_x.to(xt)
            out_y = out_y.to(at)
        if self.output_meta: return out_x, out_y, tf, {'name': orig_name[:-12], 'ao_uuid': self.label[orig_name][tf_idx][-11:-3], 'tf_pts': label['tf_pts']}
        else:                return out_x, out_y, tf


    def __len__(self):
        return len(self.items)

#%% Layers
def mish(x): return x * torch.tanh(F.softplus(x))
class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class ConvBlock(nn.Module):
    def __init__(self, n_filters, ks=3, padding=1, stride=1, bias=False, norm_cls=InstanceNorm3d, act_cls=Mish, act_first=True):
        super().__init__()
        assert isinstance(n_filters, (list, tuple)) and len(n_filters) > 1
        self.act_first = act_first
        self.convs = ModuleList([])
        self.norms = ModuleList([])
        self.act   = act_cls()
        for fin, fout in zip(n_filters, n_filters[1:]):
            self.convs.append(Conv3d(fin, fout, kernel_size=ks, padding=padding, stride=stride, bias=bias))
            if act_first: self.norms.append(norm_cls(fin))
            else:         self.norms.append(norm_cls(fout))

    def forward(self, x, tf=None):
        for conv, norm in zip(self.convs, self.norms):
            if self.act_first:     # Act > Norm > Conv
                x = self.act(x)
                if tf is None: x = norm(x)
                else:          x = norm(x, tf)
            x = conv(x)
            if not self.act_first: # Conv > Act > Norm
                if tf is None: x = norm(self.act(x))
                else:          x = norm(self.act(x), tf)
        return x

class Noop(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x, *args, **kwargs): return x

class NoAct(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x, *args, **kwargs): return x

class ResBlock(nn.Module):
    def __init__(self, fin, fh, fout, ks=3, stride=1, padding=1, bias=False, norm_cls=InstanceNorm3d, act_cls=Mish, act_first=True):
        super().__init__()
        assert isinstance(n_filters, (list, tuple)) and len(n_filters) > 1
        self.conv1 = ConvBlock([fin, fh],  ks=ks, stride=stride, padding=padding, bias=bias, norm_cls=norm_cls, act_cls=act_cls, act_first=True)
        self.conv2 = ConvBlock([fh, fout], ks=ks, stride=stride, padding=padding, bias=bias, norm_cls=norm_cls, act_cls=act_cls, act_first=True)
        self.act   = act_cls()

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='trilinear'):
        super().__init__()
        self.sf   = scale_factor
        self.mode = mode

    def forward(self, x):
        if isinstance(x, tuple):
            x, tf = x
            return F.interpolate(x, scale_factor=self.sf, mode=self.mode), tf
        else:
            return F.interpolate(x, scale_factor=self.sf, mode=self.mode)

class ConditionedInstanceNorm3d(nn.Module):
    def __init__(self, num_feature_maps, tf_desc_sz, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.var = nn.Linear(tf_desc_sz, num_feature_maps)
        self.mu  = nn.Linear(tf_desc_sz, num_feature_maps)
        nn.init.normal_(self.var.weight, std=1e-2)
        nn.init.constant_(self.var.bias, 1.0)
        nn.init.normal_(self.mu.weight,  std=1e-2)
        nn.init.constant_(self.mu.bias,  0.0)
        self.n_feature_maps = num_feature_maps

    def forward(self, x, tf_desc):
        dtype = x.dtype
        bs = x.size(0)

        mu, var = self.mu(tf_desc).view(bs, -1, 1,1,1).float(), self.var(tf_desc).view(bs, -1, 1,1,1).float()
        # xmean = x.mean(dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1)
        # xvar  = x.var( dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1)
        # return (var * (  ( (x - xmean) / (xvar + self.eps) ) + mu  )).to(dtype)
        return ((x - mu) / (var + self.eps)).to(dtype)

class MiniPointNet(nn.Module):
    def __init__(self, desc_sz, n_mid_mlps):
        super().__init__()
        self.first = nn.Linear(2, desc_sz)
        self.middle = nn.ModuleList([Mish(), nn.BatchNorm1d(desc_sz), nn.Linear(desc_sz, desc_sz)]*n_mid_mlps)
        self.last_bn = nn.BatchNorm1d(desc_sz)
        self.desc_sz = desc_sz
        self.n_mid_mlps = n_mid_mlps

    def forward(self, x):
        tf_lens = list(map(len, x))
        x = torch.cat(x, dim=0)
        x = self.first(x.view(-1, 2))
        for lin in self.middle: x = lin(x)
        x = self.last_bn(x)
        descs_per_tf = x.split(tf_lens, dim=0)
        max_fn = lambda t: t.max(dim=0).values
        return torch.stack(list(map(max_fn, descs_per_tf)))

class ExtractorConv1d(nn.Module):
    def __init__(self, desc_sz, start_nf=8, num_convs=4, pool_sz=16):
        super().__init__()
        assert num_convs > 2
        nf = start_nf
        layers = [nn.Sequential(
            Conv1d(1, nf, kernel_size=3, padding=1, stride=2),
            BatchNorm1d(nf),
            Mish()
        )]
        for _ in range(num_convs-1):
            layers.append(nn.Sequential(
                Conv1d(nf, 2*nf, kernel_size=3, padding=1, stride=2),
                BatchNorm1d(2*nf),
                Mish()
            ))
            nf *= 2
        layers.append(nn.Sequential(
            AdaptiveAvgPool1d(pool_sz),
            nn.Flatten(),
            Linear(nf * pool_sz, desc_sz)
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

def compute_local_shading(vol, light, tf_pts):
    dev = vol.device
    light  = light.to(dev)
    tf_pts = tf_pts
    bs = vol.size(0)
    light_col = light[:, :3]
    light_pos = light[:, 3:]
    tfd       = apply_tf_torch(vol, tf_pts)
    normals   = compute_normals(vol)
    orig_shape= vol.shape
    n_vox = orig_shape[-1] * orig_shape[-2] * orig_shape[-3]
    # N dot L, * light color * voxel color
    #                                       BS, XYZ, 3        BS, X*Y*Z, 3        BS, 3, 1           BS, X*Y*Z, 1 -> BS, X*Y*Z, 3 (replicated)
    local_shading = torch.matmul(normals.permute(0,2,3,4,1).view(bs, -1, 3), light_pos.unsqueeze(2)).expand(-1, -1, 3) \
        * light_col.view(bs, 1, 3).expand(-1, n_vox, 3) \
        * tfd.permute(0, 2,3,4, 1).view(bs, -1, 4)[:, :, :3]
    local_shading = local_shading.permute(0,2,1).reshape(bs, 3, *orig_shape[-3:])
    return torch.cat([local_shading, tfd[:, 3].unsqueeze(1)], dim=1)

class Injection(Enum):
    Preclassified = 0
    AdaIN = 1
    LatentCat = 2
    OmniCat = 3
    Preshaded = 4

class Extractor(Enum):
    ApplyTF = 0
    Conv1d = 1
    Pointnet = 2
    Preshade = 3

def get_extractor_from_projname(proj_name):
    if   'preclassified' in proj_name \
      or 'omnicat'       in proj_name: return Extractor.ApplyTF
    elif 'preshaded'     in proj_name: return Extractor.Preshade
    elif '1dconv'        in proj_name: return Extractor.Conv1d
    elif 'pointnet'      in proj_name: return Extractor.Pointnet
    else: raise Exception('Extractor Enum Error', f'Proj name ({proj_name}) does not contain a valid extractor keyword')

def get_injection_from_projname(proj_name):
    if   'preclassified' in proj_name: return Injection.Preclassified
    elif 'preshaded'     in proj_name: return Injection.Preshaded
    elif 'adain'         in proj_name: return Injection.AdaIN
    elif 'latentcat'     in proj_name: return Injection.LatentCat
    elif 'omnicat'       in proj_name: return Injection.OmniCat
    else: raise Exception('Injection Enum Error', f'Proj name ({proj_name}) does not contain a valid injection strategy')
#%% Model
class Unet3D(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Determine Dataset class and compute train/valid split to be used in [train/valid]_dataloader()
        self.hparams = hparams
                     # Use the dataset with random TFs
        self.items = [n for n in os.listdir(self.hparams.ds_path) if n.endswith('_original.pt')]
        self.ds_cls = QureDataset
        random.shuffle(self.items)
        self.split_idx = math.floor(len(self.items) * 0.8)

        # Determine TF Extractor & Injection from project name
        self.inj = get_injection_from_projname(self.hparams.proj_name)
        self.ext = get_extractor_from_projname(self.hparams.proj_name)
        print(f'Initializing Unet3D with Injection {self.inj} and Extractor {self.ext}')
        self.tf_desc_sz = hparams.tf_desc_sz
        if   self.ext == Extractor.Conv1d:
            self.tf_extractor = ExtractorConv1d(self.tf_desc_sz, num_convs=self.hparams.tf_extractor_depth)
            self.tf_as_pts    = False
        elif self.ext == Extractor.Pointnet:
            self.tf_extractor = MiniPointNet(self.tf_desc_sz, self.hparams.tf_extractor_depth)
            self.tf_as_pts    = True
        elif self.ext == Extractor.ApplyTF:
            self.tf_extractor = lambda tf_pts: partial(apply_tf_torch, tf_pts=tf_pts)
            self.tf_as_pts    = True
        elif self.ext == Extractor.Preshade:
            self.tf_extractor = lambda tf_pts: partial(compute_local_shading, tf_pts=tf_pts)
            self.tf_as_pts    = True

        if  self.inj == Injection.AdaIN:
            def get_norm_cls(): return partial(ConditionedInstanceNorm3d, tf_desc_sz=self.tf_desc_sz)
        else:
            def get_norm_cls(): return InstanceNorm3d

        # Set loss function
        def _ssim_fn(pred, targ, **kwargs):
            pred4d = pred.view(*pred.shape[:3], -1) if pred.ndim > 4 else pred
            return 1.0 - pt_ssim(pred4d, targ.to(pred.dtype).view(*pred4d.shape),
                data_range=1.0,
                size_average=True,
                nonnegative_ssim=True)
        def _ssim_fn_permute(pred, targ, **kwargs):
            #                     BS,C,Z, X,Y           N,      C*Z,           X,             Y
            pred4d = pred.permute(0, 1, 4,2,3).reshape(-1, pred.size(4), pred.size(2), pred.size(3)) if pred.ndim > 4 else pred
            targ4d = targ.permute(0, 1, 4,2,3).reshape(-1, targ.size(4), targ.size(2), targ.size(3)) if targ.ndim > 4 else targ
            return 1.0 - pt_ssim(pred4d, targ4d.to(pred.dtype),
                data_range=1.0,
                size_average=True,
                nonnegative_ssim=True)
        if   hparams.loss == 'huber': self.loss_fn = F.smooth_l1_loss
        elif hparams.loss == 'mse':   self.loss_fn = F.mse_loss
        elif hparams.loss == 'bce':   self.loss_fn = F.binary_cross_entropy
        elif hparams.loss == 'ssim':  self.loss_fn = _ssim_fn_permute
        elif hparams.loss == 'ssim3d':self.loss_fn = lambda p,t: 1.0 - ssim3d(p,t, win_size=self.hparams.ssim_win_sz)
        elif hparams.loss in ['mse+ssim', 'ssim+mse']:     self.loss_fn = lambda p, t: self.hparams.loss_scale_mse * F.mse_loss(p,t.to(p.dtype)) + self.hparams.loss_scale_ssim * _ssim_fn_permute(p,t)
        elif hparams.loss in ['mse+ssim3d', 'ssim3d+mse']: self.loss_fn = lambda p, t: self.hparams.loss_scale_mse * F.mse_loss(p, t.to(p.dtype)) + self.hparams.loss_scale_ssim * (1.0 - ssim3d(p, t, win_size=self.hparams.ssim_win_sz))
        else: raise Exception(f'Invalid loss function specified: {hparams.loss}')

        # Unet Layer Setup
        n_resizes = math.floor(math.log2(self.hparams.vol_sz))
        nf     = self.hparams.n_filters
        max_nf = self.hparams.max_nf
        self.first_conv = ConvBlock([1,       nf], norm_cls=Identity, act_cls=NoAct)
        self.last_conv  = ConvBlock([nf * 2 + (1 if self.inj == Injection.OmniCat else 0),  1])
        encoder, decoder, skip_szs = [], [], []
        for i in range(n_resizes):
            encoder.append(ConvBlock([min(nf, max_nf), min(nf, max_nf), min(nf*2, max_nf)], norm_cls=Noop if i+1==n_resizes else InstanceNorm3d))
            skip_szs.append(min(nf, max_nf) + (1 if self.inj == Injection.OmniCat else 0))
            nf *= 2
        skip_szs.append(min(nf, max_nf) + (1 if self.inj == Injection.OmniCat else 0))
        # Increase middle conv blocks input features to account for TF_desc concat
        if self.inj == Injection.LatentCat: self.mid_conv = ConvBlock([min(nf, max_nf)+self.tf_desc_sz, min(nf, max_nf)], norm_cls=Noop)
        else:                               self.mid_conv = ConvBlock([min(nf, max_nf), min(nf, max_nf)], norm_cls=Noop)
        for i in range(n_resizes):
            decoder.append(ConvBlock([skip_szs[-i-1]+min(nf, max_nf), min(nf, max_nf), min(nf//2, max_nf)], norm_cls=get_norm_cls()))
            nf //= 2
        self.encoder, self.decoder = ModuleList(encoder), ModuleList(decoder)

    def forward(self, x, tf):
        tf_desc = self.tf_extractor(tf) # Uses either PointNet, 1D-Convs or constructs a applicable TF
        if   self.inj == Injection.Preclassified: x = tf_desc(x) # tf_desc is a function in this case and applies the TF
        elif self.inj == Injection.OmniCat:
            op  = tf_desc(x)
            ops = [F.interpolate(op, scale_factor=2**(-i)) for i in range(len(self.encoder) + 1)]
            del op
        else: tf_desc = tf_desc.repeat_interleave(1, dim=0)
        x = self.first_conv(x)
        skips = [torch.cat([x, ops.pop(0)], dim=1)] if self.inj == Injection.OmniCat else [x]
        for layer in self.encoder:
            x = F.interpolate(x, scale_factor=0.5)
            x = layer(x)
            skips.append(torch.cat([x, ops.pop(0)], dim=1) if self.inj == Injection.OmniCat else x)
        if self.inj == Injection.LatentCat:
            x = torch.cat([x, tf_desc.view(-1, self.tf_desc_sz, 1,1,1).expand(x.size(0), -1, 1,1,1)], dim=1)
        x = self.mid_conv(x)
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1).contiguous()
        for layer in self.decoder:
            x = F.interpolate(x, size=skips[-1].shape[-3:])
            x = layer(x, tf=(tf_desc if self.inj == Injection.AdaIN else None))
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1).contiguous()
        x = self.last_conv(x)
        return x.sigmoid()

    def compute_loss(self, pred, targ, x, tf=torch.Tensor([[0.0,1.0],[1.0,1.0]])):
        return self.loss_fn(pred, targ.to(pred.dtype))

    def training_step(self, batch, batch_nb):
        x, y, tf = batch
        if not self.hparams.use_16bit:
            x, y = x.float(), y.float()
        pred = self.forward(x, tf)
        loss = self.compute_loss(pred, torch.clamp(y, 1e-8, 1- 1e-8), x, tf=tf)
        # pred4d = pred.detach().view(*pred.shape[:3], -1)
        # ssim = pt_ssim(pred4d, y.to(pred.dtype).view(*pred4d.shape), data_range=1.0, size_average=True)
        # mse  = F.mse_loss(pred.detach(), y.to(pred.dtype))
        return {
            'loss': loss,
            'log': {
                'train_loss': loss,
                # 'train_mse': mse,
                # 'train_ssim': ssim,
                # 'train_mse_ssim_ratio': mse / ssim
            }
        }

    def validation_step(self, batch, batch_nb):
        x, targ, tf = batch
        pred = self.forward(x, tf)
        targ = targ.to(pred.dtype).to(pred.device)
        loss = self.compute_loss(pred, torch.clamp(targ, 1e-8, 1- 1e-8), x, tf=tf)
        pred4d = pred.permute(0, 1, 4,2,3).reshape(-1, pred.size(4), pred.size(2), pred.size(3))
        targ4d = targ.permute(0, 1, 4,2,3).reshape(-1, targ.size(4), targ.size(2), targ.size(3))
        ssim = pt_ssim(pred4d, targ4d, data_range=1.0, size_average=True)
        ssim3 = ssim3d(pred, targ)
        mse  = F.mse_loss(pred, targ)
        return {
            'val_loss': loss,
            'val_ssim': ssim,
            'val_ssim3d': ssim3,
            'val_mse': mse,
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        avg_ssim3d = torch.stack([x['val_ssim3d'] for x in outputs]).mean()
        avg_mse  = torch.stack([x['val_mse']  for x in outputs]).mean()
        return {
            'val_loss': avg_loss,
            'log': {
                'val_loss': avg_loss,
                'val_ssim': avg_ssim,
                'val_ssim3d': avg_ssim3d,
                'val_mse':  avg_mse,
                'val_mse_ssim_ratio': avg_mse / avg_ssim
            },
            'progress_bar': {
                'val_loss': avg_loss,
                'val_ssim': avg_ssim,
                'val_ssim3d': avg_ssim3d,
                'val_mse':  avg_mse
            }
        }

    def configure_optimizers(self):
        if   self.hparams.opt.lower() == 'ranger':
            opt = Ranger(self.parameters(),              lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        elif self.hparams.opt.lower() == 'adam':
            opt = torch.optim.Adam(self.parameters(),    lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        elif self.hparams.opt.lower() == 'adamw':
            opt = torch.optim.AdamW(self.parameters(),   lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        elif self.hparams.opt.lower() == 'rmsprop':
            opt = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        elif self.hparams.opt.lower() == 'sgd':
            opt = torch.optim.SGD(self.parameters(),     lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        else:
            print(f'Invalid optimizer given: {self.hparams.opt}')
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, min_lr=1e-8)
        return [opt], [sch]

    def collate_fn(self, batch):
        # Qure Dataset returns tuple (intensity vol, ao vol, TF)
        # Where TF is either a list of points or a texture.
        x = torch.cat([it[0] for it in batch], dim=0)
        y = torch.cat([it[1] for it in batch], dim=0)
        if   self.tf_as_pts: #
            tf = [it[2] for it in batch]
        else:    # Stack textures
            tf = torch.stack([it[2] for it in batch])
        return x, y, tf

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            dataset=self.ds_cls(self.hparams.ds_path,
                items=self.items[:self.split_idx],
                tf_as_pts=self.tf_as_pts,
                vol_sz=self.hparams.vol_sz,
                device=torch.device('cpu')
            ),
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            dataset=self.ds_cls(self.hparams.ds_path,
                items=self.items[self.split_idx:],
                tf_as_pts=self.tf_as_pts,
                vol_sz=self.hparams.vol_sz,
                device=torch.device('cpu')
            ),
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

    def add_model_specific_args(parser, root_dir):
        parser.add_argument('ds_path',                         type=str,   help='Folder containing CQ500 dataset (output of cuda_runner.py)')
        parser.add_argument('--n_filters',      default=16,    type=int,   help='Number of filters for first conv. Subsequent convs double the number of filters.')
        parser.add_argument('--max_nf',         default=512,   type=int,   help='Max number of filters in Unet. (may exceed with concats like latentcat)')
        parser.add_argument('--learning_rate',  default=1e-3,  type=float, help='Max Learning rate for the Ranger (RAdam + LookAhead) Optimizer')
        parser.add_argument('--wd',             default=0.0,   type=float, help='Weight decay used during training')
        parser.add_argument('--batch_size',     default=1,     type=int,   help='Number of volumes per batch')
        parser.add_argument('--max_epochs',     default=50,    type=int,   help='Max number of epochs')
        parser.add_argument('--min_epochs',     default=10,    type=int,   help='Min number of epochs')
        parser.add_argument('--tf_desc_sz',     default=128,    type=int,   help='Size of the transfer function descriptor coming out of the point net')
        parser.add_argument('--tf_extractor_depth', default=3, type=int,   help='Depth of the TF extractor (either Mini PointNet or Conv1dExtractor)')
        parser.add_argument('--vol_sz',        default=64,    type=int,   help='Size of input volume tiles')
        parser.add_argument('--loss_scale_mse', default=5.0,   type=float, help='Scales the MSE loss term')
        parser.add_argument('--loss_scale_ssim',default=1.0,   type=float, help='Scales the DSSIM loss term')
        parser.add_argument('--ssim_win_sz',    default=11,    type=int,   help='Size of 3D SSIM window')
        parser.add_argument('--loss',        default='mse+ssim',  type=str,   help='Loss function to use. Possible is huber, mse, bce, ssim, mse+ssim')
        parser.add_argument('--opt',         default='ranger', type=str,   help='Which optimizer is used. Options are Ranger, Adam, AdamW, RMSprop, SGD (case insensitive)')
        return parser

def main(hparams):
    if not hparams.seed:
        hparams.seed = np.random.randint(int(2**32)-1)
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    model = Unet3D(hparams)


    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        amp_level='O2',
        show_progress_bar=True,
        row_log_interval=10,
        log_save_interval=10,
        log_gpu_memory=None,
        weights_summary=None,
        print_nan_grads=hparams.track_grads,
        track_grad_norm=2 if hparams.track_grads else -1,
        fast_dev_run=hparams.debug_run,
        overfit_pct=hparams.overfit_pct,
        val_check_interval=hparams.val_every,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=hparams.accumulate_grads,

        early_stop_callback=pl.callbacks.EarlyStopping('val_loss', patience=5),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=f'{os.getcwd()}/lightning_logs/{hparams.proj_name}/checkpoints/',
            save_top_k=2
        )
    )

    trainer.fit(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser()

    parent_parser.add_argument('proj_name',             type=str,                 help='Experiment name for Comet.ML')
    parent_parser.add_argument('--gpus',                type=int,   default=1,    help='Number of GPUs used for training')
    parent_parser.add_argument('--distributed_backend', type=str,   default=None, help='One of dp, ddp, ddp2')
    parent_parser.add_argument('--accumulate_grads',    type=int,   default=1,    help='Number of batches for Gradient Accumulation')
    parent_parser.add_argument('--val_every',           type=float, default=1,    help='Do a validation run every n epochs. (can be float)')
    parent_parser.add_argument('--overfit_pct',         type=float, default=0.0,  help='Lets the model overfit on this percentage of the train data')
    parent_parser.add_argument('--seed',                type=int,   default=None, help='Sets random seed for NumPy and PyTorch')
    parent_parser.add_argument('--debug_run', dest='debug_run', action='store_true',     help='Only run 1 train, valid, test batch for debugging.')
    parent_parser.add_argument('--track_grads', dest='track_grads', action='store_true', help='Whether gradients are tracked')
    parent_parser.add_argument('--all_fp32', dest='use_16bit', action='store_false',     help='Disable AMP and fall back to full 32 bit float training')

    parser = Unet3D.add_model_specific_args(parent_parser, root_dir)
    hps = parser.parse_args()
    main(hps)


# %%
