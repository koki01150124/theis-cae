"""エントロピー符号化・復号の整合性の検証．"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "programs", "ex001_kodak_cae"))
import entropy_coding as ec  # noqa: E402
from model import CAE  # noqa: E402


def _laplace_ints(shape, scale, seed):
    """ラプラス分布に従う整数配列を作る．

    Args:
        shape (tuple): 配列の形状．
        scale (float): ラプラス分布のスケール．
        seed (int): 乱数 seed．
    Returns:
        np.ndarray: 整数配列．
    """
    return np.round(np.random.default_rng(seed).laplace(0, scale, size=shape)).astype(np.int64)


def test_roundtrip_and_rate():
    """符号化・復号で係数が完全に一致し，符号長が理想符号長とほぼ一致することを確認する．"""
    k = 6
    train = _laplace_ints((50, k, 16, 16), 3.0, 0)
    hist = ec.HistogramModel(k)
    hist.update(train)
    hist.finalize()
    z = _laplace_ints((k, 64, 96), 3.0, 1)
    z[0, 0, 0] = 500  # 学習範囲外（エスケープ）
    z[3, 5, 7] = -999
    stream, payload = ec.encode(z, hist, 512, 768, model_id=2, setting_id=7)
    z_dec, info = ec.decode(stream, hist)
    assert np.array_equal(z, z_dec)
    assert info == {"height": 512, "width": 768, "model_id": 2, "setting_id": 7}
    ideal = hist.ideal_bits(z)
    assert payload * 8 <= ideal * 1.01 + 64
    assert payload * 8 >= ideal * 0.99


def test_state_dict_roundtrip():
    """確率モデルの保存・復元後も同じビットストリームが得られることを確認する．"""
    hist = ec.HistogramModel(3)
    hist.update(_laplace_ints((10, 3, 8, 8), 2.0, 2))
    hist.finalize()
    hist2 = ec.HistogramModel.from_state_dict(hist.state_dict())
    z = _laplace_ints((3, 8, 8), 2.0, 3)
    assert ec.encode(z, hist, 64, 64)[0] == ec.encode(z, hist2, 64, 64)[0]


def test_model_pipeline_exact():
    """モデルの encode -> 符号化 -> 復号 -> decode の再構成がビットストリームを経ない場合と一致することを確認する．"""
    torch.manual_seed(0)
    model = CAE(16).eval()
    with torch.no_grad():
        model.log_scales.fill_(1.0)
        x = torch.rand(1, 3, 64, 96) * 255
        z = model.encode(x)
        hist = ec.HistogramModel(16)
        hist.update(z[0].numpy())
        hist.finalize()
        stream, _ = ec.encode(z[0].numpy().astype(np.int64), hist, 64, 96)
        z_dec, _ = ec.decode(stream, hist)
        rec_direct = model.decode(z)
        rec_stream = model.decode(torch.from_numpy(z_dec).float()[None])
    assert torch.equal(rec_direct, rec_stream)
