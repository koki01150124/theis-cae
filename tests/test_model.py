"""CAE のモデル・損失・Incremental Training の学習前検証．"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "programs", "ex001_kodak_cae"))
from model import CAE, GSM, clip_ste, round_ste  # noqa: E402
from train import IncrementalController, loss_func  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _smooth_images(batch, size, seed=0):
    """滑らかな合成画像（低周波の正弦波の和）を作る．

    Args:
        batch (int): 枚数．
        size (int): 一辺の画素数．
        seed (int): 乱数 seed．
    Returns:
        torch.Tensor: 形状 (batch, 3, size, size)，値域 0〜255．
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    imgs = torch.zeros(batch, 3, size, size)
    for _ in range(4):
        f = torch.rand(batch, 3, 2, generator=g) * 6
        p = torch.rand(batch, 3, 1, 1, generator=g) * 6.28
        imgs += torch.sin(f[..., 0, None, None] * 6.28 * xx + f[..., 1, None, None] * 6.28 * yy + p)
    return (imgs - imgs.amin((2, 3), keepdim=True)) / (imgs.amax((2, 3), keepdim=True) - imgs.amin((2, 3), keepdim=True)) * 255


@pytest.mark.parametrize("h,w,ch", [(128, 128, 96), (512, 768, 96), (768, 512, 64)])
def test_shapes(h, w, ch):
    """入出力形状と潜在表現の形状（1/8 解像度）が一致することを確認する．"""
    model = CAE(ch).to(DEVICE)
    x = torch.rand(2, 3, h, w, device=DEVICE) * 255
    x_hat, log_q, z = model(x)
    assert x_hat.shape == x.shape
    assert z.shape == (2, ch, h // 8, w // 8)
    assert log_q.shape == z.shape
    with torch.no_grad():
        assert model.decode(model.encode(x)).shape == x.shape


def test_round_ste_gradient():
    """丸めの順伝播が round，逆伝播の勾配が 1 であることを確認する．"""
    y = torch.tensor([-1.6, -0.4, 0.49, 0.51, 2.5, 3.7], requires_grad=True)
    out = round_ste(y)
    assert torch.equal(out, torch.round(y.detach()))
    (out * torch.arange(6.0)).sum().backward()
    assert torch.allclose(y.grad, torch.arange(6.0))


def test_clip_ste_gradient():
    """クリップの順伝播が [0, 255]，範囲外でも勾配が 1 であることを確認する．"""
    x = torch.tensor([-50.0, 0.0, 100.0, 255.0, 400.0], requires_grad=True)
    out = clip_ste(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 100.0, 255.0, 255.0]))
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones(5))


def test_gsm_numerical_stability():
    """GSM の対数密度が極端な値・パラメータでも有限で，勾配も有限であることを確認する．"""
    gsm = GSM(4)
    with torch.no_grad():
        gsm.log_precisions[0] = torch.tensor([-30.0, -10.0, 0.0, 10.0, 30.0, 40.0])
        gsm.log_weights[1] = torch.tensor([-100.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    v = torch.tensor([0.0, 1e-8, 0.5, 50.0, 1e4, -1e4]).view(1, 1, 1, 6).repeat(1, 4, 1, 1).requires_grad_()
    lp = gsm.log_prob(v)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    assert torch.isfinite(v.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in gsm.parameters())
    assert torch.isfinite(gsm.discrete_log2_prob(torch.round(v.detach()))).all()


def test_gsm_normalization():
    """GSM の密度の積分と離散確率 Q(z) の総和がいずれも 1 になることを確認する．"""
    gsm = GSM(2).double()
    v = torch.linspace(-200, 200, 400001, dtype=torch.float64).view(1, 1, 1, -1).repeat(1, 2, 1, 1)
    integral = torch.exp(gsm.log_prob(v)).sum(-1) * (v[0, 0, 0, 1] - v[0, 0, 0, 0])
    assert torch.allclose(integral, torch.ones_like(integral), atol=1e-3)
    z = torch.arange(-200, 201.0, dtype=torch.float64).view(1, 1, 1, -1).repeat(1, 2, 1, 1)
    total = (2.0 ** gsm.discrete_log2_prob(z).double()).sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_loss_terms():
    """式 (16) の各項（レート項・歪み項）が定義どおりに計算されることを確認する．"""
    x = torch.rand(3, 3, 16, 16) * 255
    x_hat = torch.rand(3, 3, 16, 16) * 255
    log_q = -torch.rand(3, 8, 2, 2) * 5
    z = torch.zeros(3, 8, 2, 2)
    for alpha in [0.01, 0.05, 0.2]:
        out = loss_func((x_hat, log_q, z), x, alpha)
        rate = (-log_q.flatten(1).sum(1) / 32).mean()
        dist = (((x - x_hat) ** 2).flatten(1).sum(1) / (1000 * 768)).mean()
        assert math.isclose(out["rate_term"].item(), rate.item(), rel_tol=1e-6)
        assert math.isclose(out["dist_term"].item(), dist.item(), rel_tol=1e-6)
        assert math.isclose(out["loss"].item(), alpha * rate.item() + (1 - alpha) * dist.item(), rel_tol=1e-6)


def test_mask_zeroes_disabled_channels():
    """マスクで無効化したチャネルの係数が 0 になることを確認する．"""
    model = CAE(8)
    model.mask.copy_(torch.tensor([1, 1, 0, 0, 0, 0, 0, 0.0]))
    _, _, z = model(torch.rand(1, 3, 64, 64) * 255)
    assert (z[:, 2:] == 0).all()


def test_scale_parameters_roundtrip():
    """スケール λ を変えると量子化の粒度が変わり，decode で λ が除されることを確認する．"""
    model = CAE(8).eval()
    x = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        y = model.encoder(model.normalize(x))
        model.log_scales.fill_(math.log(4.0))
        assert torch.equal(model.encode(x), torch.round(y * 4.0))


def test_incremental_controller():
    """改善が続く間は有効化せず，停滞したら 1 チャネルずつ有効化することを確認する．"""
    ctrl = IncrementalController(5, initial=2, window=10, threshold=0.01, wait=20)
    step = 0
    for i in range(100):  # 毎窓 10% 改善
        step += 1
        assert not ctrl.step(1.0 * 0.99 ** i, step)
    assert ctrl.enabled == 2
    added = sum(ctrl.step(0.5, step + i) for i in range(1, 200))
    assert added >= 3 and ctrl.enabled == 5 and ctrl.complete


def test_overfit_small_batch():
    """小規模データ（固定の 4 枚）に過学習させ，損失が大きく下がり PSNR が上がることを確認する．"""
    torch.manual_seed(0)
    model = CAE(96, mean=(127.0,) * 3, std=(60.0,) * 3).to(DEVICE)
    x = _smooth_images(4, 64).to(DEVICE)
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "log_scales"], lr=1e-3)
    first = None
    for _ in range(400):
        out = model(x)
        loss = loss_func(out, x, 0.01)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    with torch.no_grad():
        mse = ((model.decode(model.encode(x)) - x) ** 2).mean().item()
    assert loss.item() < 0.3 * first
    assert 10 * math.log10(255 ** 2 / mse) > 28.0
