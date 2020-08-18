import torch
import torch.nn.functional as F

def gaussian3d(device=torch.device('cpu'), dtype=torch.float32):
    gauss2d = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]], device=device).to(dtype) / 16.0
    return torch.stack([gauss2d, 2*gauss2d, gauss2d]) / 4.0

def get_gaussian1d(size, sigma, dtype=torch.float32):
    coords = torch.arange(size)
    coords-= size//2

    gauss = torch.exp(- coords**2 / (2*sigma**2))
    gauss/= gauss.sum()
    return gauss.to(dtype)

def filter_gaussian_separated(input, win):
    win = win.to(input.dtype).to(input.device)
    out = F.conv3d(input, win,                groups=input.size(1))
    out = F.conv3d(out,   win.transpose(3,4), groups=input.size(1))
    out = F.conv3d(out,   win.transpose(2,4), groups=input.size(1))
    return out

def ssim3d(pred, targ, data_range=1.0, win_size=11, sigma=1.5, non_negative=True, return_average=True):
    N, C, W, H, D = pred.shape
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2

    # win = gaussian3d(device=pred.device, dtype=pred.dtype)[None,None]
    win = get_gaussian1d(win_size, sigma, dtype=pred.dtype).to(pred.device)[None, None, None, None]
    mu1, mu2 = filter_gaussian_separated(pred, win), filter_gaussian_separated(targ, win)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_gaussian_separated(pred * pred, win) - mu1_sq
    sigma2_sq = filter_gaussian_separated(targ * targ, win) - mu2_sq
    sigma12   = filter_gaussian_separated(pred * targ, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if non_negative: cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if return_average: return ssim_map.mean()
    else:              return ssim_map

def ssim1d(pred, targ, data_range=1.0, win_size=11, sigma=1.5, non_negative=True, return_average=True):
    N, C, W, H, D = pred.shape
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2

    # win = gaussian3d(device=pred.device, dtype=pred.dtype)[None,None]
    win = get_gaussian1d(win_size, sigma, dtype=pred.dtype).to(pred.device)[None, None]
    mu1, mu2 = F.conv1d(pred, win, groups=pred.size(1)), F.conv1d(targ, win, groups=pred.size(1))

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv1d(pred * pred, win, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv1d(targ * targ, win, groups=targ.size(1)) - mu2_sq
    sigma12   = F.conv1d(pred * targ, win, groups=pred.size(1)) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if non_negative: cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if return_average: return ssim_map.mean()
    else:              return ssim_map
