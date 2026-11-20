"""式 (9) のスケールパラメータ λ の微調整（論文 2.3 節，3.2 節）．

学習済みの CAE について，λ 以外の全パラメータ（Encoder，Decoder，GSM）を固定し，
目標の α ごとにチャネル単位の log-scale を最適化する．
    - 損失: 式 (16)（目標の α を用いる）
    - 学習率: 初期値 1e-3 に τ^κ / (τ + t)^κ を掛けて連続的に減衰（κ = 0.8，τ = 1000）
    - 更新数: 10,000，batch size 32
推定仕様（論文に記載なし）:
    - 目標 α の集合 FINETUNE_ALPHAS（3 モデル共通）．論文 Figure 9 の各モデルのレート範囲を覆うよう選んだ．
      当初は 0.002〜0.7 の 9 通りとしたが，最も情報量の多い画像でも論文と同程度の低レート（約 0.36 bpp）まで
      下げられるよう 0.85 と 0.95 を追加した．
    - log-scale の初期値は 0（λ = 1）．

出力: outputs/ex001_kodak_cae/CAE/{α}_{ch}/{seed}/scales/alpha_{target}.json
実行: .venv_pytorch/bin/python programs/ex001_kodak_cae/finetune_scales.py
"""
import argparse
import itertools
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import load_dataloader, load_stats  # noqa: E402
from model import CAE, load_model  # noqa: E402
from train import BATCH_SIZE, MODEL_CONFIGS, OUTPUT_DIR, SEEDS, loss_func, run_parallel  # noqa: E402

FINETUNE_ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.85, 0.95]
FINETUNE_ITERS = 10000
LR0 = 1e-3
KAPPA = 0.8
TAU = 1000.0


def model_dir(alpha, channels, seed=0):
    """学習済みモデルのディレクトリを返す．

    Args:
        alpha (float): 学習時の α．
        channels (int): 潜在チャネル数．
        seed (int): seed．
    Returns:
        str: ディレクトリのパス．
    """
    return os.path.join(OUTPUT_DIR, "CAE", f"{alpha}_{channels}", str(seed))


def load_trained(alpha, channels, seed=0, weights="final_model.pth", device="cuda"):
    """学習済み CAE を読み込む．final_model.pth が無い場合は checkpoint.pth の重みを使う．

    Args:
        alpha (float): 学習時の α．
        channels (int): 潜在チャネル数．
        seed (int): seed．
        weights (str): 重みファイル名．
        device (str): 配置先．
    Returns:
        tuple[CAE, str]: (モデル, 使用した重みファイルのパス)．
    """
    d = model_dir(alpha, channels, seed)
    path = os.path.join(d, weights)
    if not os.path.exists(path):
        path = os.path.join(d, "checkpoint.pth")
    stats = load_stats()
    model = load_model(CAE, path, seed, {"latent_channels": channels, "mean": stats["mean"], "std": stats["std"]})
    return model.to(device).eval(), path


def finetune(model, target_alpha, loader, iters=FINETUNE_ITERS, position=0):
    """λ のみを目標 α で最適化する．

    Args:
        model (CAE): 学習済みモデル（λ 以外は固定される）．
        target_alpha (float): 目標の α．
        loader (torch.utils.data.DataLoader): 学習用 DataLoader．
        iters (int): 更新数．
        position (int): tqdm の表示位置．
    Returns:
        dict: {"log_scales": list, "history": list[[step, loss, rate, dist]]}．
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.log_scales.data.zero_()
    model.log_scales.requires_grad_(True)
    opt = torch.optim.Adam([model.log_scales], lr=LR0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: TAU ** KAPPA / (TAU + t) ** KAPPA)
    history, step = [], 0
    bar = tqdm(total=iters, leave=False, position=position, desc=f"alpha={target_alpha}")
    while step < iters:
        for x, _ in loader:
            x = x.cuda(non_blocking=True)
            out = loss_func(model(x), x, target_alpha)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            sched.step()
            step += 1
            bar.update(1)
            if step % 100 == 0:
                history.append([step, out["loss"].item(), out["rate_term"].item(), out["dist_term"].item()])
            if step >= iters:
                break
    bar.close()
    model.log_scales.requires_grad_(False)
    return {"log_scales": model.log_scales.detach().cpu().tolist(), "history": history}


def run_model(args):
    """1 モデルについて全ての目標 α で λ を微調整する（既に結果がある α はスキップ）．

    Args:
        args (tuple): (alpha, channels, seed, position)．
    Returns:
        str: 出力ディレクトリ．
    """
    alpha, channels, seed, position = args
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, path = load_trained(alpha, channels, seed)
    out_dir = os.path.join(model_dir(alpha, channels, seed), "scales")
    os.makedirs(out_dir, exist_ok=True)
    loader, _ = load_dataloader(seed, BATCH_SIZE, samples_per_epoch=BATCH_SIZE * 1000, num_workers=4)
    for target in FINETUNE_ALPHAS:
        out_path = os.path.join(out_dir, f"alpha_{target}.json")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        result = finetune(model, target, loader, position=position)
        result.update({"target_alpha": target, "base_alpha": alpha, "channels": channels, "weights": path,
                       "iters": FINETUNE_ITERS, "lr0": LR0, "kappa": KAPPA, "tau": TAU, "elapsed_sec": time.time() - t0})
        with open(out_path, "w") as f:
            json.dump(result, f)
        last = result["history"][-1]
        tqdm.write(f"[{alpha}_{channels}] alpha={target}: loss {last[1]:.5f} rate {last[2]:.4f} dist {last[3]:.5f} "
                   f"mean log-scale {sum(result['log_scales']) / channels:.3f}")
    return out_dir


def main():
    """全モデルの λ を並列に微調整する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="例: 0.2_64")
    args = parser.parse_args()
    jobs = [(a, c, s, i) for i, ((a, c), s) in enumerate(itertools.product(MODEL_CONFIGS, SEEDS))
            if args.only in (None, f"{a}_{c}")]
    print("終了コード:", run_parallel(run_model, jobs, len(jobs)))


if __name__ == "__main__":
    main()
