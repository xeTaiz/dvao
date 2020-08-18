#%% Init
import os
import numpy as np
import ctypes
import argparse
import math
from ctypes import *
from pathlib import Path
from volume_loader import *
from utils import *
from tf_utils import *
import time
from uuid import uuid4
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F

def get_cuda_raycast_volume():
    dll = ctypes.CDLL('/home/dome/Repositories/deep-volume-rendering/raycast_volume.so', mode=ctypes.RTLD_GLOBAL)
    raycastVolume = dll.raycastVolume
    raycastVolume.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_int32), POINTER(c_float),
    POINTER(c_float), c_float, c_size_t, c_float, POINTER(c_float)]

    return raycastVolume

_cuda_raycast_volume = get_cuda_raycast_volume()

def raycast_volume(vol, tf_tex, tex_dims, vox_scl, ray, n_steps_factor, n_rays, min_value, vol_out):
    vol_p     = vol.ctypes.data_as(      POINTER(c_float))
    tf_tex_p  = tf_tex.ctypes.data_as(   POINTER(c_float))
    tex_dims_p= tex_dims.ctypes.data_as( POINTER(c_int32))
    vox_scl_p = vox_scl.ctypes.data_as(  POINTER(c_float))
    ray_p     = ray.ctypes.data_as(      POINTER(c_float))
    vol_out_p = vol_out.ctypes.data_as(  POINTER(c_float))
    _cuda_raycast_volume(vol_p, tf_tex_p, tex_dims_p, vox_scl_p, ray_p, n_steps_factor, n_rays, min_value, vol_out_p)

def generate_random_rays(n_rays):
    theta, phi = np.random.rand(2, n_rays)
    theta *= 2.0 * np.pi
    phi = np.arccos(phi * 2.0 - 1.0)

    return np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ], axis=1)

def generate_uniform_rays(samples=256,randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.stack(points)

def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def get_gradient(vol):
    device = vol.device
    dtype = vol.dtype
    gauss2d = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]], device=device).to(dtype)
    sobel3d_x = torch.stack([gauss2d, 0*gauss2d, -gauss2d])
    sobel3d_y = sobel3d_x.permute(1,0,2)[None, None] # make 5D tensor, because conv3d expects 5D
    sobel3d_z = sobel3d_x.permute(1,2,0)[None, None]
    sobel3d_x = sobel3d_x[None, None]

    grad = torch.stack([F.conv3d(vol, sobel3d_x), F.conv3d(vol, sobel3d_y), F.conv3d(vol, sobel3d_z)], dim=-1)
    return grad.squeeze()

def normalize_inviwo(vol, data_range, value_range):
    return (vol - data_range[0]) / (data_range[1] - data_range[0]) * (value_range[1] - value_range[0]) + value_range[0]

def normalize_hounsfield(vol):
    if vol.max() > 1.0:
        ret = vol.copy()
        if vol.min() < 0: ret[vol == vol.min()] = 0.0
        return np.clip(ret / 4095.0, 0.0, 1.0).astype(np.float32)
    else: return vol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raycast Volume for AO')
    parser.add_argument('-nrays', metavar='-N', type=int, default=256, required=False, help='Number of rays to cast per voxel')
    parser.add_argument('-stepfactor', metavar='-R', type=float, default=1.0, required=False, help='Fraction to divide nSteps by')
    parser.add_argument('--uniform', action='store_true', help='Use uniformly distributed rays using Spherical Fibonacci')
    parser.add_argument('--debug', action='store_true', help='Saves all 4 channels of the volume for debugging')
    parser.add_argument('--only1', action='store_true', help='Process only the first DICOM folder')
    parser.add_argument('--random_tf', action='store_true', help='Whether to generate a random TF every time')
    parser.add_argument('--tf_path', type=str, default='transfer_functions/qure_bone_new.itf', help='Path to the .itf transfer function')
    parser.add_argument('--tf_res', type=int, default=4096, help='Transfer function resolution')
    parser.add_argument('--vol_path', type=str, default='/run/media/dome/Data/data/QureAI_Brain_CT', help='Path to the QureAI CQ500 dataset')
    parser.add_argument('--out_path', type=str, default='/run/media/dome/Data/data/Volumes/Qure_RandomTF', help='Path where the resulting AO shall be saved')
    parser.add_argument('--pt', action='store_true', help='Whether the vol_path contains NumPy volumes in PyTorch pickled dicts (*.pt)')
    parser.add_argument('--skip_existing', action='store_true', help='Skips Volumes for which the target file already exists')
    args = parser.parse_args()
#%% Run
    path = Path(args.vol_path)

    if args.random_tf: # Generate random TF in volume_generator()
        tf_pts = None
    else:              # Load TF from tf_path
        if not args.pt: # Read TFs from tf_path
            tf_path = Path(args.tf_path)
            tf_pts = read_inviwo_tf(tf_path)
            tf_tex = make_tf_tex_from_pts(tf_pts, args.tf_res)

    if args.pt: # Load NumpyArrays, also
        vol_list = path.rglob('*.pt')
        def _gen():
            for vol_p in vol_list:
                data = torch.load(path/vol_p)
                vol     = np.asfortranarray(data['vol'].astype(np.float32))
                tf_pts  = data['tf_pts'].astype(np.float32)
                name    = data['vol_name']
                vox_scl = data['vox_scl'].astype(np.float32)
                yield vol, tf_pts, vox_scl, name
        volume_gen = _gen()
    else: # Load DICOM folders using CQ500 directory structure
        volume_dirs = list(
            filter(lambda p: p is not None,
            map(   get_largest_dir,                             # extract subdir with most files in it (highest res volume)
            map(   lambda p: list(p.iterdir()),                 # get list of actual volume directorie
            map(   lambda p: next(p.iterdir())/'Unknown Study', # cd into subfolders CQ500-CT-XX/Unknown Study/
            filter(lambda p: p.is_dir(),                        # Get all dirs, no files
            path.iterdir())))))                                  # Iterate over path directory
        )

        volume_gen = get_volume_gen(volume_dirs, False, tf_pts=tf_pts)

    for vol, tf_pts, voxel_scale, vol_name in volume_gen:
        tf_tex = make_tf_tex_from_pts(tf_pts, resolution=args.tf_res)
        save_path = Path(args.out_path)
        orig_name = f'{vol_name}_original.pt'
        save_name = save_path/f'{vol_name}_{str(uuid4())[:8]}.pt'
        files_no_uuid = filter(lambda p: p[:-12], os.listdir(save_path))
        if vol_name in files_no_uuid and args.skip_existing: continue
        # Get scaling variables, normalize volume
        tex_dims = np.array([*vol.shape, args.tf_res]).astype(np.int32)
        n_rays = args.nrays
        n_steps_factor = args.stepfactor
        # Generate Rays
        if args.uniform: rays = generate_uniform_rays(n_rays).astype(np.float32)
        else:            rays = generate_random_rays(n_rays).astype(np.float32)
        # Pass to CUDA
        cont_vol = np.ascontiguousarray(normalize_hounsfield(vol).transpose(2,1,0).reshape(-1))
        out = np.zeros(4*cont_vol.size, dtype=np.float32)
        raycast_volume(cont_vol, tf_tex, tex_dims, voxel_scale, rays, n_steps_factor, n_rays, cont_vol.min(), out)
        out = out.reshape([*reversed(vol.shape), 4]).transpose(2,1,0,3)
        # Save result
        out_ao  = np.ascontiguousarray(out[:,:,:,3] / n_rays)

        torch.save({
            'vol_path': str(save_path/orig_name),
            'ao':     torch.from_numpy(out_ao).half(),
            'tf_pts': torch.from_numpy(tf_pts),
            'tf_tex': torch.from_numpy(tf_tex),
            'vox_scl':torch.from_numpy(voxel_scale)
        }, save_name)
        if not (save_path/orig_name).exists(): # save input volume if not present in save_path
            out_vol = np.ascontiguousarray(out[:,:,:,1].astype(np.float16))
            torch.save(torch.from_numpy(out_vol), save_path/orig_name)
        if args.debug:                         # save all channels
            np.save(save_path/f'debug_{args.suffix}.npy', np.ascontiguousarray(out.astype(np.float16)))

        print(f'Saved AO ({out_ao.min(), out_ao.max()}({n_rays} rays, uniform={args.uniform}): {save_name}.')
        if args.only1: break


def run_raycast(vol, tf_pts, voxel_scale, n_rays=512, n_steps_factor=0.1, uniform=True, tf_res=4096):
    ''' Computes raycast ambient occlusion volume
    Args:
        vol (ndarray): Volume of shape (X,Y,Z) with data range [0,1]
        tf_pts (ndarray): Shape (N, C) for N points with C-1 channels
        voxel_scale (ndarray): vec3 repreenting voxel scale with min == 1.0
        n_rays (int, optional): Number of rays. Defaults to 512.
        n_steps_factor (float, optional): Ray length will be `n_steps_factor * diag(vol)`. Defaults to 0.1.
        uniform (bool, optional): Use uniform ray distribution. When `False` use random rays. Defaults to True.
        tf_res (int, optional): TF texture resolution. Defaults to 4096.

    Returns:
        [ndarray]: Ambient occlusion volume of shape (X,Y,Z) in range [0,1]
    '''
    tf_tex = make_tf_tex_from_pts(tf_pts, resolution=tf_res)
    tex_dims = np.array([*vol.shape, tf_res]).astype(np.int32)

    if uniform: rays = generate_uniform_rays(n_rays).astype(np.float32)
    else:       rays = generate_random_rays(n_rays).astype(np.float32)

    cont_vol = np.ascontiguousarray(vol.reshape(-1))
    out = np.zeros(4*vol.size, dtype=np.float32)
    print('vol.shape: ', vol.shape)
    t0 = time.time()
    raycast_volume(cont_vol.astype(np.float32), tf_tex, tex_dims, voxel_scale, rays, n_steps_factor, n_rays, cont_vol.min(), out)
    print(f'Volumetric AO Raycasting took {time.time() - t0}s.')
    out = out.reshape([*reversed(vol.shape), 4]).transpose(2,1,0,3)
    print('out.shape: ', out.shape)
    return np.ascontiguousarray(out[:,:,:,3] / n_rays)
