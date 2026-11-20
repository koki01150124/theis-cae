"""画質評価指標（PSNR，SSIM，MS-SSIM）．

論文 3.3 節に合わせ，SSIM は scikit-image (van der Walt et al., 2014) の実装を，
MS-SSIM は Toderici et al. (2016b) の実装
(https://github.com/tensorflow/models/blob/2390974a/compression/msssim.py, Apache-2.0) を
TensorFlow 非依存に移植したものを用いる．いずれも RGB，画素値 0〜255 で計算する．
SSIM の設定（カラー/グレースケール，窓）は論文に記載がないため，scikit-image の既定値
（7x7 一様窓，RGB 各チャネルの平均，data_range=255）を用いる．
"""
import numpy as np
from scipy import signal
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity


def psnr(img1, img2):
    """PSNR [dB] を計算する．

    Args:
        img1 (np.ndarray): 形状 (H, W, 3) の uint8 原画像．
        img2 (np.ndarray): 形状 (H, W, 3) の uint8 再構成画像．
    Returns:
        float: PSNR [dB]．
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def ssim(img1, img2):
    """scikit-image による SSIM（RGB，既定の窓）を計算する．

    Args:
        img1 (np.ndarray): 形状 (H, W, 3) の uint8 原画像．
        img2 (np.ndarray): 形状 (H, W, 3) の uint8 再構成画像．
    Returns:
        float: SSIM．
    """
    return float(structural_similarity(img1, img2, channel_axis=-1, data_range=255))


def ssim_gray(img1, img2):
    """輝度（ITU-R BT.601）に変換した画像の SSIM を計算する（評価定義の比較用）．

    Args:
        img1 (np.ndarray): 形状 (H, W, 3) の uint8 原画像．
        img2 (np.ndarray): 形状 (H, W, 3) の uint8 再構成画像．
    Returns:
        float: SSIM．
    """
    w = np.array([0.299, 0.587, 0.114])
    return float(structural_similarity(img1.astype(np.float64) @ w, img2.astype(np.float64) @ w, data_range=255))


def _fspecial_gauss(size, sigma):
    """MATLAB の fspecial('gaussian') を模したガウス窓を返す．

    Args:
        size (int): 窓の一辺．
        sigma (float): 標準偏差．
    Returns:
        np.ndarray: 形状 (size, size) の正規化済み窓．
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _ssim_for_multiscale(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """1 スケール分の SSIM と contrast-structure を計算する（Toderici et al. の実装の移植）．

    Args:
        img1 (np.ndarray): 形状 (1, H, W, 3) の float64 画像．
        img2 (np.ndarray): 形状 (1, H, W, 3) の float64 画像．
        max_val (float): 画素値のダイナミックレンジ．
        filter_size (int): ガウス窓の一辺．
        filter_sigma (float): ガウス窓の標準偏差．
        k1 (float): 安定化定数．
        k2 (float): 安定化定数．
    Returns:
        tuple[float, float]: (平均 SSIM, 平均 contrast-structure)．
    """
    _, height, width, _ = img1.shape
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = np.reshape(_fspecial_gauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode="valid")
    mu2 = signal.fftconvolve(img2, window, mode="valid")
    sigma11 = signal.fftconvolve(img1 * img1, window, mode="valid") - mu1 * mu1
    sigma22 = signal.fftconvolve(img2 * img2, window, mode="valid") - mu2 * mu2
    sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid") - mu1 * mu2
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim_val = np.mean(((2.0 * mu1 * mu2 + c1) * v1) / ((mu1 * mu1 + mu2 * mu2 + c1) * v2))
    return ssim_val, np.mean(v1 / v2)


def msssim(img1, img2, max_val=255):
    """Toderici et al. (2016b) の実装と同一の MS-SSIM を計算する．

    Args:
        img1 (np.ndarray): 形状 (H, W, 3) の uint8 原画像．
        img2 (np.ndarray): 形状 (H, W, 3) の uint8 再構成画像．
        max_val (float): 画素値のダイナミックレンジ．
    Returns:
        float: MS-SSIM．
    """
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1 = img1[None].astype(np.float64)
    im2 = img2[None].astype(np.float64)
    mssim, mcs = [], []
    for _ in range(levels):
        s, cs = _ssim_for_multiscale(im1, im2, max_val=max_val)
        mssim.append(s)
        mcs.append(cs)
        im1, im2 = [convolve(im, downsample_filter, mode="reflect")[:, ::2, ::2, :] for im in (im1, im2)]
    mssim, mcs = np.array(mssim), np.array(mcs)
    return float(np.prod(mcs[:levels - 1] ** weights[:levels - 1]) * (mssim[levels - 1] ** weights[levels - 1]))


def all_metrics(img1, img2):
    """PSNR，SSIM，MS-SSIM をまとめて計算する．

    Args:
        img1 (np.ndarray): 形状 (H, W, 3) の uint8 原画像．
        img2 (np.ndarray): 形状 (H, W, 3) の uint8 再構成画像．
    Returns:
        dict[str, float]: {"psnr", "ssim", "msssim"}．
    """
    return {"psnr": psnr(img1, img2), "ssim": ssim(img1, img2), "msssim": msssim(img1, img2)}
