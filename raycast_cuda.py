import os, ctypes, math
from ctypes import *
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from xml.dom import minidom

from volume_loader import *
from utils import *
from tqdm import tqdm, trange


def get_cuda_raycast_volume():
    dll = ctypes.CDLL('./raycast_volume.so', mode=ctypes.RTLD_GLOBAL)
    raycastVolume = dll.raycastVolume
    raycastVolume.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_int32), POINTER(c_float),
    POINTER(c_float), c_float, c_size_t, POINTER(c_float)]

    return raycastVolume

_cuda_raycast_volume = get_cuda_raycast_volume()

def raycast_volume(vol, tf_tex, tex_dims, vox_scl, ray, n_steps_factor, n_rays, vol_out):
    vol_p     = vol.ctypes.data_as(      POINTER(c_float))
    tf_tex_p  = tf_tex.ctypes.data_as(   POINTER(c_float))
    tex_dims_p= tex_dims.ctypes.data_as( POINTER(c_int32))
    vox_scl_p = vox_scl.ctypes.data_as(  POINTER(c_float))
    ray_p     = ray.ctypes.data_as(      POINTER(c_float))
    vol_out_p = vol_out.ctypes.data_as(  POINTER(c_float))
    _cuda_raycast_volume(vol_p, tf_tex_p, tex_dims_p, vox_scl_p, ray_p, n_steps_factor, n_rays, vol_out_p)

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

def read_inviwo_tf(fn):
    xmldoc = minidom.parse(str(fn))
    def parse_point(point):
        pos = float(point.getElementsByTagName('pos')[0].getAttribute('content'))
        opacity = float(point.getElementsByTagName('rgba')[0].getAttribute('w'))
        return pos, opacity
    points = sorted(map(parse_point, xmldoc.getElementsByTagName('Point')))
    l, r = points[0][1], points[-1][1]
    xp, yp = zip(*points)
    def apply_tf(x, normalize=False):
        if normalize: x = normalize_minmax(x)
        return np.interp(x, xp, yp, left=l, right=r)

    return apply_tf

def make_tf_texture(tf, resolution=1024):
    return tf(np.linspace(0, 1, resolution, dtype=np.float32)).astype(np.float32)

def normalize_hounsfield(vol):
    ret = vol + 0
    if vol.min() < 0: ret[vol == vol.min()] = 4095
    return np.clip(ret / 4095.0, 0.0, 1.0)


if __name__ == '__main__':
    parser = ArgumentParser(description='Python wrapper for CUDA AO Raycaster')
    parser.add_argument('out_path', type=str, help='Directory where the raycasted results are saved.')
    parser.add_argument('-n', '--n_rays', type=int, default=256, help='Number of rays to cast per voxel')
    parser.add_argument('-s', '--step_factor', type=float, default=1.0, help='Fraction to divide nSteps by')
    parser.add_argument('--uniform', action='store_true', help='Use uniformly distributed rays using Spherical Fibonacci')
    parser.add_argument('--only1', action='store_true', help='Process only the first DICOM folder')
    parser.add_argument('--suffix', type=str, default=None, help='Attaches passed suffix to saved file name')
    parser.add_argument('--tf_path', type=str, default='transfer_functions/qure_bone_new.itf', help='Path to the .itf transfer function')
    parser.add_argument('--vol_path', type=str, default='/run/media/dome/Data/data/Volumes/QureNpy', help='Path to the volumes as .npy')
    parser.add_argument('--tf_res', type=int, default=4096, help='Resolution of the 1D transfer function texture')
    parser.add_argument('--skip_existing', action='store_true', help='Skips Volumes for which the target file already exists')
    args = parser.parse_args()
    # Paths
    vol_path = Path(args.vol_path)
    tf_path  = Path(args.tf_path)
    vol_fns  = os.listdir(vol_path)
    out_path = Path(args.out_path)
    # Transfer Function
    transfer_function = read_inviwo_tf(tf_path)
    tf_tex = make_tf_texture(transfer_function, args.tf_res)
    # Process
    for vol_fn in tqdm(vol_fns):
        # Load Volume
        vol = np.load(vol_path/vol_fn)
        cont_vol = np.ascontiguousarray(vol.transpose(2,1,0))
        # Parse volume specific params
        tex_dims = np.array([*vol.shape, args.tf_res]).astype(np.int32)
        s_idx = vol_fn.find('_', 5) + 1          # Read 7 character float from filename,
        vox_z_scl = float(vol_fn[s_idx:s_idx+7]) # starting from 2nd '_', first _ is always at 4
        vol_name = vol_fn[5:s_idx] # Read volume name between first two '_'
        voxel_scale = np.array([1, 1, vox_z_scl], dtype=np.float32)
        complete_save_path = out_path/f'ao_{vol_name}{"_"+args.suffix if args.suffix else ""}.npy'
        if args.skip_existing and complete_save_path.exists(): continue # Skip if exists
        # Generate Rays
        if args.uniform: rays = generate_uniform_rays(args.n_rays).astype(np.float32)
        else:            rays = generate_random_rays(args.n_rays).astype(np.float32)
        # Launch Raycaster
        out = np.zeros(4 * cont_vol.size, dtype=np.float32)
        t0 = time.time()
        raycast_volume(cont_vol, tf_tex, tex_dims, voxel_scale, rays, args.step_factor, args.n_rays, out)
        print(f'Raycasting took {time.time() - t0}s.')
        out = out.reshape([*reversed(vol.shape), 4]).transpose(2,1,0,3)
        ao = np.ascontiguousarray(out[:,:,:, 3].astype(np.float16) / args.n_rays)
        it = np.ascontiguousarray(out[:,:,:, 1].astype(np.float16))
        # Check whether the output intensity is the same as input intensity
        # assert np.sqrt((vol - it)**2).max() < 1e-3, "Raycaster produced very different Intensity! Weird memory alignment"
        # Save Results
        np.save(complete_save_path, ao)
        if args.only1: break # Only raycast 1 volume
