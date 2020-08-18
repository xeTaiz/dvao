import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from xml.dom   import minidom

from peak_finder import get_persistent_homology
from torchinterp1d import Interp1d


def read_inviwo_tf(fn):
    xmldoc = minidom.parse(str(fn))
    def parse_point(point):
        pos = float(point.getElementsByTagName('pos')[0].getAttribute('content'))
        opacity = float(point.getElementsByTagName('rgba')[0].getAttribute('w'))
        return pos, opacity
    points = sorted(map(parse_point, xmldoc.getElementsByTagName('Point')))
    l, r = points[0][1], points[-1][1]
    xp, yp = zip(*points)

    return np.array(points)

def color_generator():
    ''' Generates distinguishable colors, compare
    http://alumni.media.mit.edu/~wad/color/numbers.html
    '''
    colors = np.array([
        [173, 35, 35],
        [42, 75, 215],
        [29, 105, 20],
        [129, 74, 25],
        [129, 38, 192],
        [160, 160, 160],
        [129, 197, 122],
        [157, 175, 255],
        [41, 208, 208],
        [255, 146, 51],
        [255, 238, 51],
        [255, 205, 243],
        [255, 255, 255]
    ], dtype=np.float32) / 255.0
    for color in colors:
        yield color

def normalize_hounsfield(vol, value):
    ''' Moves the negative number peak to `value` (if negatives exist), then normalizes by 4096 and clips [0,1] '''
    ret = vol.copy()
    if vol.min() < 0: ret[vol == vol.min()] = value
    return np.clip(ret / 4095.0, 0.0, 1.0)

def apply_box_tf(x, value=0.4, lb=1560.0, ub=3060.0):
    out = np.zeros_like(x) + value
    out[x < lb] = 0.0
    out[x > ub] = 0.0
    return out

def apply_gaussian(x, mean, std, magnitude):
    return magnitude * np.exp(- (x-mean)**2 / (2*std**2))

def make_tf_texture(tf, resolution=4096):
    return tf(np.linspace(0, 1, resolution, dtype=np.float32)).astype(np.float32)

def make_tf_tex_from_pts(tf_pts, resolution=4096):
    nc = tf_pts.shape[1]
    return np.interp(np.linspace(0, 1, resolution, dtype=np.float32), tf_pts[:, 0], tf_pts[:, -1]).astype(np.float32)

def make_tf_tex_from_pts_torch(tf_pts, resolution=4096):
    return apply_tf_torch(torch.linspace(0.0, 1.0, resolution)[None,None], tf_pts)

def apply_tf_torch(x, tf_pts):
    ''' Applies the TF described by points `tf_pts` (N x [0,1]^C+1 with x pos and C channels) to `x
    Args:
        x (torch.Tensor): The intensity values to apply the TF on. Assumed shape is (N, 1, ...) with batch size N
        tf_pts (torch.Tensor): Tensor of shape (N, (1+C)) containing N points consisting of x coordinate and mapped features (RGBO)
    Returns:
        torch.Tensor: Tensor with TF applied of shape (N, C, ...) with batch size N (same as `x`) and number of channels C (same as `tf_pts`)
    '''
    if isinstance(tf_pts, list): return torch.cat([apply_tf_torch(x, tf) for tf in tf_pts], dim=0) # If tf_pts is in a list, perform for each item in that list
    dev = x.device
    tf_pts = tf_pts.to(dev)
    npt, nc = tf_pts.shape
    x_shap = tuple(x.shape)
    x_out_shap = (x_shap[0], nc-1, *x_shap[2:])
    x_acc   = torch.empty(x_out_shap,        dtype=torch.float32, device=dev)
    pts_acc = torch.empty((npt * (nc-1), 2), dtype=torch.float32, device=dev)
    for i in range(1,nc):
        x_acc[:, i-1] = x + (i-1) # make intensity volume of shape (N, nc, W,H,D), with intensity values offset by 1 for each channel
        pts_acc[(i-1)*npt:i*npt] = tf_pts[:, [0,i]] + torch.Tensor([[i-1,0.0]]).to(dev) # offset TF values (xRGBO) similarly to get all channels aligned to intensity [0, nc-1]
    return Interp1d()(pts_acc[:,0].float(), pts_acc[:,1].float(), x_acc.float().view(-1)).reshape(x_out_shap).to(x.dtype) # Interp on flattened volume, reshape

def get_histogram_peaks(data, bins=1000, skip_outlier=True):
    if skip_outlier:
        if data.min() < 0.0 or data.max() > 1.0:
            data = normalize_hounsfield(data)
        vals, ranges = np.histogram(data[np.logical_and(data < 1.0, data > 1e-2)], bins)
    else:
        vals, ranges = np.histogram(data, bins)
    peaks = get_persistent_homology(vals)
    ret = np.array(list(map(lambda p: (
        (ranges[p.born] + ranges[p.born+1])/2.0,    # intensity value
        p.get_persistence(vals)), peaks # persistence for peak importance
        )))
    return np.stack([ret[:, 0], ret[:, 1] / peaks[0].get_persistence(vals)], axis=1)

def overlaps_trapeze(trap, ts):
    for t in ts:
        if trap[0,0] < t[5,0] and trap[5,0] > t[0,0]: return True
    return False

def includes_maxvalue(trap, vol=None):
    return trap[5, 0] >= (1.0 if vol is None else vol.max())

def includes_minvalue(trap, vol=None, eps=1e-2):
    return trap[0, 0] <= (eps if vol is None else vol.min() + eps)

def colorize_trapeze(t, color):
    res = np.zeros((t.shape[0], 5))
    res[:, 0]   = t[:, 0]
    res[:, 1:4] = color
    res[:, 4]   = t[:, 1]
    return res

def get_trapezoid_tf_points_from_peaks(peaks, height_range=(0.1, 0.9), width_range=(0.02, 0.2), max_num_peaks=5):
    ''' Compute transfer function with trapezoids around given peaks
    Args:
        peaks (np.array of [intensity, persistence]): The histogram peaks
        height_range (tuple of floats): Range in which to draw trapezoid height (=opacity). Max range is (0, 1)
        width_range (tuple of floats): Range in which to draw trapezoid width around peak. Max range is (0, 1)
        max_num_peaks (int): Maximum number of peaks in the histogram. The number will be drawn as U(1, max_num_peaks)
    Returns:
        [ np.array [x, y] ]: List of TF primitives (List of coordinates [0,1]²) to be lerped
    '''
    num_peaks = np.random.randint(1, max_num_peaks)
    height_range_len = height_range[1] - height_range[0]
    width_range_len  = width_range[1] - width_range[0]
    color_gen = color_generator()
    def make_trapezoid(c, top_height, bot_width):
        bot_height = np.random.rand(1).item() * top_height
        top_width  = np.random.rand(1).item() * bot_width
        return np.stack([
          np.array([c - bot_width/2 -1e-2, 0]),    # left wall          ____________  __ top_height
          np.array([c - bot_width/2, bot_height]), # bottom left       / top_width  \
          np.array([c - top_width/2, top_height]), # top left        /__ bot_width __\__ bot_height
          np.array([c + top_width/2, top_height]), # top right      |                |
          np.array([c + bot_width/2, bot_height]), # bottom right   |   right wall ->|
          np.array([c + bot_width/2 +1e-2, 0])     # right wall     |<- left wall    |
        ])                                         #               |        c       |__ 0

    trapezes = [make_trapezoid(c, # Center of peak
        top_height= height_range_len * np.random.rand(1).item() + height_range[0],
        bot_width = width_range_len  * np.random.rand(1).item() + width_range[0]
        ) for c, p in peaks]
    result = []
    for t in trapezes:
        if overlaps_trapeze(t, result) or includes_maxvalue(t) or includes_minvalue(t): continue
        else: result.append(colorize_trapeze(t, next(color_gen)))
        if len(result) >= max_num_peaks: break
    res_arr = np.stack(result)
    np.random.shuffle(res_arr)
    res_arr = np.clip(res_arr[:num_peaks].reshape((-1, 5)), 0, 1)
    idx = np.argsort(res_arr[:, 0])
    return res_arr[idx]

def get_tf_fn_from_peaks(peaks, mean_noise=0, magnitude_mean=0.5, magnitude_std=0.0, min_std=0.1):
    tf_fns = [
        partial(apply_gaussian,
            mean=p[0] + mean_noise * np.random.randn(1),                   # Add noise around mean
            std=1/max(p[1], min_std) * (1.5* np.random.rand(1) + 0.5),     # scale std by with z ~ U(0.5, 2.0)
            magnitude=magnitude_std * np.random.randn(1) + magnitude_mean) # magnitude is N(mag_mean, mag_std)
        for p in peaks ]
    return np.sum(np.stack([tf_fn(np.linspace(0, 4095, 4096)) for tf_fn in tf_fns]), axis=0)

def compute_normals(vol):
    device = vol.device
    dtype = vol.dtype
    gauss2d = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]], device=device).to(dtype) / 16.0
    sobel3d_x = torch.stack([gauss2d, 0*gauss2d, -gauss2d])
    sobel3d_y = sobel3d_x.permute(1,0,2)[None, None] # make 5D tensor, because conv3d expects 5D
    sobel3d_z = sobel3d_x.permute(1,2,0)[None, None]
    sobel3d_x = sobel3d_x[None, None]
    vol = F.pad(vol, (1,1,1,1,1,1), mode='replicate')
    grad = torch.cat([F.conv3d(vol, sobel3d_x), F.conv3d(vol, sobel3d_y), F.conv3d(vol, sobel3d_z)], dim=1)
    grad_lens = torch.norm(grad, dim=1, keepdim=True)
    grad_lens[grad_lens < 1e-4] = 1
    return grad / grad_lens
