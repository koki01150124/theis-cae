"""チェックポイントの保存と学習再開の検証（合成データで短い学習を行う）．"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "programs", "ex001_kodak_cae"))
import train as train_module  # noqa: E402
from model import CAE  # noqa: E402


def _fake_loader(seed=0, batch_size=4, samples_per_epoch=16, num_workers=0):
    """合成データの DataLoader を返す（load_dataloader と同じ引数・戻り値）．

    Args:
        seed (int): 乱数 seed．
        batch_size (int): ミニバッチサイズ．
        samples_per_epoch (int): 1 エポックのサンプル数．
        num_workers (int): 未使用．
    Returns:
        tuple: (学習用 DataLoader, 検証用 DataLoader)．
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(samples_per_epoch, 3, 64, 64, generator=g) * 255
    ds = torch.utils.data.TensorDataset(x, x)
    return (torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g),
            torch.utils.data.DataLoader(ds, batch_size=batch_size))


def test_checkpoint_resume(tmp_path, monkeypatch):
    """2 エポック学習後に 4 エポックまで再開でき，状態（更新数・ロガー・最適化状態）が引き継がれることを確認する．"""
    monkeypatch.setattr(train_module, "ITERS_PER_EPOCH", 4)
    monkeypatch.setattr(train_module, "load_stats", lambda: {"mean": [120.0] * 3, "std": [60.0] * 3})
    hp = {"alpha": 0.05, "channels": 8}
    target = str(tmp_path / "run")
    train_module.train(target, CAE, _fake_loader, 2, 4, 0, hp)
    ckpt = torch.load(os.path.join(target, "checkpoint.pth"), weights_only=False)
    assert ckpt["state"]["epoch"] == 2 and ckpt["state"]["global_step"] == 8
    os.remove(os.path.join(target, "finished.json"))
    train_module.train(target, CAE, _fake_loader, 4, 4, 0, hp)
    ckpt2 = torch.load(os.path.join(target, "checkpoint.pth"), weights_only=False)
    assert ckpt2["state"]["epoch"] == 4 and ckpt2["state"]["global_step"] == 16
    with open(os.path.join(target, "log.json")) as f:
        log = json.load(f)
    assert log["epoch"] == [0, 1, 2, 3, 4]
    assert ckpt2["optimizer"]["state"][0]["step"].item() == 16
    for name in ["best_model.pth", "final_model.pth", "config.json", "iter_log.csv", "train_stdout.log"]:
        assert os.path.exists(os.path.join(target, name))
