"""Theis et al. (ICLR 2017) Compressive Autoencoder (CAE) のモデル定義．

論文 Figure 2 に基づく Encoder / Decoder，Residual Block，Sub-pixel Convolution，
整数量子化の Straight-Through Estimator (STE)，クリッピングの勾配近似，
Gaussian Scale Mixture (GSM) によるエントロピーモデルを実装する．

論文に明記された仕様:
    - Encoder: normalize -> mirror-pad -> conv 64x5x5/2 -> conv 128x5x5/2
      -> Residual Block x3 (128x3x3 x2) -> conv Cx5x5/2 -> round
    - Decoder: subpix 512x3x3/2 -> Residual Block x3 -> subpix 256x3x3/2
      -> subpix 12x3x3/2 -> denormalize -> clip(0, 255)
    - Encoder は valid 畳み込み + ミラーパディング，Decoder は zero padding 畳み込み
    - 丸めの微分は恒等写像 (式 (3), r(y)=y)，クリップの微分は 1 (式 (14))
    - GSM は各チャネル独立で 6 スケール，log-weight と log-precision で最適化
    - スケールパラメータ λ はチャネルごと（空間方向で共有），log-scale で最適化
推定仕様（論文に記載なし）:
    - LeakyReLU の傾き 0.01（Lasagne の LeakyRectify の既定値）
    - 重み初期化 Glorot uniform，バイアス 0（Lasagne の既定値）
    - ミラーパディング幅 35 px（全層 valid 畳み込みで出力が入力の 1/8 になる値）
    - Residual Block の skip 接続は valid 畳み込みで縮んだ分を中央で切り出す
    - GSM の初期標準偏差は 0.2〜20 の対数等間隔
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import set_seed  # noqa: E402

LEAKY_SLOPE = 0.01
MIRROR_PAD = 35
NUM_GSM_SCALES = 6


# ---------------------------------------------------------------------------
# 勾配を近似する演算
# ---------------------------------------------------------------------------
class _RoundSTE(torch.autograd.Function):
    """順伝播で最近接整数への丸め，逆伝播で恒等写像の微分を用いる演算．"""

    @staticmethod
    def forward(ctx, x):
        """丸めを行う．

        Args:
            x (torch.Tensor): 任意形状の実数テンソル．
        Returns:
            torch.Tensor: x と同形状の丸められたテンソル．
        """
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        """勾配をそのまま通す (式 (3), r(y)=y)．

        Args:
            grad_output (torch.Tensor): 出力に対する勾配．
        Returns:
            torch.Tensor: 入力に対する勾配（grad_output と同一）．
        """
        return grad_output


class _ClipSTE(torch.autograd.Function):
    """順伝播で [0, 255] へのクリップ，逆伝播で微分を 1 とする演算 (式 (14))．"""

    @staticmethod
    def forward(ctx, x):
        """クリップを行う．

        Args:
            x (torch.Tensor): 任意形状の実数テンソル．
        Returns:
            torch.Tensor: x と同形状で [0, 255] に収めたテンソル．
        """
        return torch.clamp(x, 0.0, 255.0)

    @staticmethod
    def backward(ctx, grad_output):
        """勾配をそのまま通す．

        Args:
            grad_output (torch.Tensor): 出力に対する勾配．
        Returns:
            torch.Tensor: 入力に対する勾配（grad_output と同一）．
        """
        return grad_output


def round_ste(x):
    """STE 付きの丸め．

    Args:
        x (torch.Tensor): 任意形状の実数テンソル．
    Returns:
        torch.Tensor: x と同形状の丸められたテンソル．
    """
    return _RoundSTE.apply(x)


def clip_ste(x):
    """勾配 1 のクリップ（0〜255）．

    Args:
        x (torch.Tensor): 任意形状の実数テンソル．
    Returns:
        torch.Tensor: x と同形状のクリップ済みテンソル．
    """
    return _ClipSTE.apply(x)


# ---------------------------------------------------------------------------
# ネットワークの構成要素
# ---------------------------------------------------------------------------
def _init_conv(conv):
    """Lasagne の既定値に合わせて Glorot uniform で重みを，0 でバイアスを初期化する．

    Args:
        conv (nn.Conv2d): 初期化する畳み込み層．
    Returns:
        nn.Conv2d: 初期化済みの畳み込み層（同一オブジェクト）．
    """
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class ResidualBlock(nn.Module):
    """conv(LeakyReLU) -> conv(線形) -> 入力との和 からなる Residual Block．

    Encoder では valid 畳み込みのため，skip 接続側を中央で切り出して形状を合わせる．
    """

    def __init__(self, channels=128, padding=0):
        """層を構築する．

        Args:
            channels (int): 入出力チャネル数．
            padding (int): 畳み込みのゼロパディング幅（Encoder は 0，Decoder は 1）．
        """
        super().__init__()
        self.conv1 = _init_conv(nn.Conv2d(channels, channels, 3, padding=padding))
        self.conv2 = _init_conv(nn.Conv2d(channels, channels, 3, padding=padding))

    def forward(self, x):
        """Residual Block の順伝播．

        Args:
            x (torch.Tensor): 形状 (B, C, H, W) の特徴マップ．
        Returns:
            torch.Tensor: 形状 (B, C, H', W') の特徴マップ．valid の場合 H'=H-4．
        """
        h = F.leaky_relu(self.conv1(x), LEAKY_SLOPE)
        h = self.conv2(h)
        crop = (x.shape[-1] - h.shape[-1]) // 2
        if crop > 0:
            x = x[..., crop:-crop, crop:-crop]
        return x + h


class SubPixelConv(nn.Module):
    """畳み込みの後に係数を再配置 (pixel shuffle) して空間解像度を上げる層．"""

    def __init__(self, in_channels, conv_channels, upscale=2):
        """層を構築する．

        Args:
            in_channels (int): 入力チャネル数．
            conv_channels (int): 畳み込みの出力チャネル数（論文表記の C）．
            upscale (int): 拡大率．出力チャネル数は conv_channels / upscale^2．
        """
        super().__init__()
        self.conv = _init_conv(nn.Conv2d(in_channels, conv_channels, 3, padding=1))
        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        """Sub-pixel convolution の順伝播．

        Args:
            x (torch.Tensor): 形状 (B, C_in, H, W)．
        Returns:
            torch.Tensor: 形状 (B, conv_channels / 4, 2H, 2W)．
        """
        return self.shuffle(self.conv(x))


class Encoder(nn.Module):
    """CAE の Encoder f（量子化前の連続値を出力する）．"""

    def __init__(self, latent_channels=96):
        """層を構築する．

        Args:
            latent_channels (int): 潜在表現のチャネル数（96 または 64）．
        """
        super().__init__()
        self.conv1 = _init_conv(nn.Conv2d(3, 64, 5, stride=2))
        self.conv2 = _init_conv(nn.Conv2d(64, 128, 5, stride=2))
        self.res = nn.ModuleList([ResidualBlock(128, padding=0) for _ in range(3)])
        self.conv3 = _init_conv(nn.Conv2d(128, latent_channels, 5, stride=2))

    def forward(self, x_norm):
        """Encoder の順伝播．

        Args:
            x_norm (torch.Tensor): 正規化済み画像，形状 (B, 3, H, W)．H, W は 8 の倍数．
        Returns:
            torch.Tensor: 連続値の係数，形状 (B, latent_channels, H/8, W/8)．
        """
        h = F.pad(x_norm, (MIRROR_PAD,) * 4, mode="reflect")
        h = F.leaky_relu(self.conv1(h), LEAKY_SLOPE)
        h = F.leaky_relu(self.conv2(h), LEAKY_SLOPE)
        for block in self.res:
            h = block(h)
        return self.conv3(h)


class Decoder(nn.Module):
    """CAE の Decoder g（正規化空間の画像を出力する）．"""

    def __init__(self, latent_channels=96):
        """層を構築する．

        Args:
            latent_channels (int): 潜在表現のチャネル数（96 または 64）．
        """
        super().__init__()
        self.sub1 = SubPixelConv(latent_channels, 512)
        self.res = nn.ModuleList([ResidualBlock(128, padding=1) for _ in range(3)])
        self.sub2 = SubPixelConv(128, 256)
        self.sub3 = SubPixelConv(64, 12)

    def forward(self, z):
        """Decoder の順伝播．

        Args:
            z (torch.Tensor): 量子化された係数，形状 (B, latent_channels, h, w)．
        Returns:
            torch.Tensor: 正規化空間の再構成画像，形状 (B, 3, 8h, 8w)．
        """
        h = self.sub1(z)
        for block in self.res:
            h = block(h)
        h = F.leaky_relu(self.sub2(h), LEAKY_SLOPE)
        return self.sub3(h)


class GSM(nn.Module):
    """チャネルごとに独立な Gaussian Scale Mixture による係数の確率密度モデル (式 (12))．"""

    def __init__(self, channels, num_scales=NUM_GSM_SCALES):
        """パラメータを初期化する．

        Args:
            channels (int): 係数のチャネル数 K．
            num_scales (int): 混合数 S（論文では 6）．
        """
        super().__init__()
        sigmas = torch.logspace(math.log10(0.2), math.log10(20.0), num_scales)
        self.log_weights = nn.Parameter(torch.zeros(channels, num_scales))
        self.log_precisions = nn.Parameter((-2.0 * torch.log(sigmas)).repeat(channels, 1))

    def _expand(self, param, ndim):
        """(K, S) のパラメータを (1, K, 1, 1, S) などに整形する．

        Args:
            param (torch.Tensor): 形状 (K, S)．
            ndim (int): 係数テンソルの次元数（4 を想定）．
        Returns:
            torch.Tensor: 形状 (1, K, 1, ..., 1, S)．
        """
        shape = [1, param.shape[0]] + [1] * (ndim - 2) + [param.shape[1]]
        return param.view(*shape)

    def log_prob(self, v):
        """連続値 v における対数密度 ln q(v)（自然対数）を要素ごとに計算する．

        Args:
            v (torch.Tensor): 形状 (B, K, H, W)．
        Returns:
            torch.Tensor: 形状 (B, K, H, W) の ln q(v)．
        """
        log_pi = self._expand(torch.log_softmax(self.log_weights, dim=1), v.ndim)
        log_prec = self._expand(self.log_precisions, v.ndim)
        v = v.unsqueeze(-1)
        comp = log_pi + 0.5 * log_prec - 0.5 * math.log(2 * math.pi) - 0.5 * torch.exp(log_prec) * v * v
        return torch.logsumexp(comp, dim=-1)

    def discrete_log2_prob(self, z):
        """整数 z の離散確率 Q(z) = ∫_{[z-.5, z+.5)} q の log2 を計算する（推定 bpp 用）．

        Args:
            z (torch.Tensor): 整数値の係数，形状 (B, K, H, W)．
        Returns:
            torch.Tensor: 形状 (B, K, H, W) の log2 Q(z)．
        """
        pi = self._expand(torch.softmax(self.log_weights, dim=1), z.ndim).double()
        std = self._expand(torch.exp(-0.5 * self.log_precisions), z.ndim).double()
        z = z.unsqueeze(-1).double()
        upper = torch.special.ndtr((z + 0.5) / std)
        lower = torch.special.ndtr((z - 0.5) / std)
        prob = (pi * (upper - lower)).sum(-1).clamp_min(1e-300)
        return torch.log2(prob).float()


class CAE(nn.Module):
    """Compressive Autoencoder．Encoder f，Decoder g，GSM q，スケール λ，マスク m を持つ．"""

    def __init__(self, latent_channels=96, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        """モデルを構築する．

        Args:
            latent_channels (int): 潜在表現のチャネル数．
            mean (tuple[float]): 正規化に用いるチャネルごとの平均（0〜255 の画素値空間）．
            std (tuple[float]): 正規化に用いるチャネルごとの標準偏差．
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        self.gsm = GSM(latent_channels)
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("mask", torch.ones(latent_channels))
        # スケール λ = exp(log_scales)．基本学習では 0（λ=1）で固定し，式 (9) の微調整時のみ学習する．
        self.log_scales = nn.Parameter(torch.zeros(latent_channels), requires_grad=False)

    def normalize(self, x):
        """画素値を固定の平均・標準偏差で正規化する．

        Args:
            x (torch.Tensor): 形状 (B, 3, H, W)，値域 0〜255．
        Returns:
            torch.Tensor: 形状 (B, 3, H, W) の正規化画像．
        """
        return (x - self.pixel_mean) / self.pixel_std

    def denormalize(self, x):
        """正規化を元に戻す．

        Args:
            x (torch.Tensor): 形状 (B, 3, H, W) の正規化空間の画像．
        Returns:
            torch.Tensor: 形状 (B, 3, H, W) の画素値空間の画像．
        """
        return x * self.pixel_std + self.pixel_mean

    def _channel_view(self, vec):
        """(K,) のベクトルを (1, K, 1, 1) に整形する．

        Args:
            vec (torch.Tensor): 形状 (K,)．
        Returns:
            torch.Tensor: 形状 (1, K, 1, 1)．
        """
        return vec.view(1, -1, 1, 1)

    def encode(self, x):
        """画像を量子化済みの整数係数 [f(x)∘λ]∘m に変換する（評価時用）．

        Args:
            x (torch.Tensor): 形状 (B, 3, H, W)，値域 0〜255．
        Returns:
            torch.Tensor: 形状 (B, K, H/8, W/8) の整数値（float 型）の係数．
        """
        y = self.encoder(self.normalize(x)) * self._channel_view(torch.exp(self.log_scales))
        return torch.round(y) * self._channel_view(self.mask)

    def decode(self, z):
        """整数係数から画像を再構成する（評価時用）．

        Args:
            z (torch.Tensor): 形状 (B, K, h, w) の整数値の係数．
        Returns:
            torch.Tensor: 形状 (B, 3, 8h, 8w)，値域 0〜255 にクリップした再構成画像．
        """
        z = z / self._channel_view(torch.exp(self.log_scales))
        return torch.clamp(self.denormalize(self.decoder(z)), 0.0, 255.0)

    def forward(self, x):
        """学習用の順伝播．式 (9)，(13) の各項に必要な量を返す．

        Args:
            x (torch.Tensor): 形状 (B, 3, H, W)，値域 0〜255 の画像．
        Returns:
            tuple:
                x_hat (torch.Tensor): 形状 (B, 3, H, W)，クリップ済み再構成画像．
                log_q (torch.Tensor): 形状 (B, K, H/8, W/8)，ln q([f(x)∘λ]∘m + u)．
                z (torch.Tensor): 形状 (B, K, H/8, W/8)，量子化済み係数．
        """
        scales = self._channel_view(torch.exp(self.log_scales))
        y = self.encoder(self.normalize(x)) * scales
        z = round_ste(y) * self._channel_view(self.mask)
        u = torch.rand_like(z) - 0.5
        log_q = self.gsm.log_prob(z + u)
        x_hat = clip_ste(self.denormalize(self.decoder(z / scales)))
        return x_hat, log_q, z


def load_model(ModelClass, weight_path=None, seed=0, model_kwargs=None):
    """モデルをインスタンス化し，必要に応じて重みを読み込む．

    Args:
        ModelClass (type): torch.nn.Module のクラス（CAE）．
        weight_path (str | None): 読み込む重み（state_dict または学習チェックポイント）のパス．
        seed (int): パラメータ初期化の乱数 seed．
        model_kwargs (dict | None): ModelClass に渡す引数．
    Returns:
        torch.nn.Module: インスタンス化したモデル．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = ModelClass(**(model_kwargs or {}))
    if weight_path is not None:
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state)
    return model
