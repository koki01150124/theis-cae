"""学習済み CAE の Kodak 24 枚による評価（実測 bpp，推定 bpp，PSNR，SSIM，MS-SSIM）．

各モデルについて次のレート設定を評価する．
    - base          : λ = 1（学習時の α のまま）
    - alpha_{a}     : 式 (9) で目標 α=a に微調整した λ
    - interp_{a}_{b}: 隣接する 2 つの微調整済み log-scale の中点（論文 2.3 節の補間）
各設定について，学習データ (DIV2K train) の量子化係数から Laplace 平滑化ヒストグラムを作り，
レンジコーダで実際にビットストリームを生成・復号して再構成する．復号した係数が符号化前と
一致することを毎回検証する．

bpp の定義（分母は RGB 画像の画素数 H×W）:
    - bpp          : (payload のバイト数 + アンサンブル選択の 1 byte) × 8 / (H×W)   … 論文と同じ数え方
    - bpp_file     : ビットストリームファイル全体（6 byte のヘッダ含む）× 8 / (H×W)
    - bpp_est_gsm  : GSM の離散確率 Q(z) による理想符号長 / (H×W)（参考値）
    - bpp_est_hist : ヒストグラムモデルによる理想符号長 / (H×W)（参考値）

出力: outputs/ex001_kodak_cae/CAE/{α}_{ch}/{seed}/kodak/
    per_image.csv, summary.csv, histograms/{setting}.json, bitstreams/{setting}/*.bin,
    reconstructions/{setting}/*.png
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import entropy_coding as ec  # noqa: E402
from data import PackedImages, load_kodak  # noqa: E402
from finetune_scales import FINETUNE_ALPHAS, load_trained, model_dir  # noqa: E402
from metrics import all_metrics  # noqa: E402
from train import MODEL_CONFIGS  # noqa: E402

ENSEMBLE_BYTES = 1
HIST_MAX_IMAGES = 800


def rate_settings(directory):
    """評価するレート設定（名前と log-scale）を列挙する．

    Args:
        directory (str): モデルのディレクトリ．
    Returns:
        list[tuple[str, float | None, list[float] | None]]: (設定名, 目標 α, log-scale) のリスト．
            base は log-scale が None（λ=1）．
    """
    settings = [("base", None, None)]
    tuned = []
    for a in FINETUNE_ALPHAS:
        path = os.path.join(directory, "scales", f"alpha_{a}.json")
        if os.path.exists(path):
            with open(path) as f:
                tuned.append((a, json.load(f)["log_scales"]))
    for a, s in tuned:
        settings.append((f"alpha_{a}", a, s))
    for (a, s), (b, t) in zip(tuned[:-1], tuned[1:]):
        settings.append((f"interp_{a}_{b}", math.sqrt(a * b), [(u + v) / 2 for u, v in zip(s, t)]))
    return settings


@torch.no_grad()
def build_histogram(model, num_images=HIST_MAX_IMAGES):
    """学習データ全体の量子化係数からヒストグラムモデルを作る．

    Args:
        model (CAE): 評価対象のモデル（λ 設定済み）．
        num_images (int): 用いる学習画像の最大枚数．
    Returns:
        ec.HistogramModel: Laplace 平滑化済みの確率モデル．
    """
    images = PackedImages("train")
    hist = ec.HistogramModel(model.latent_channels)
    for i in tqdm(range(min(num_images, len(images))), desc="histogram", leave=False):
        img = np.array(images.image(i))
        h, w = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
        x = torch.from_numpy(img[:h, :w]).permute(2, 0, 1)[None].float().cuda()
        hist.update(model.encode(x)[0].cpu().numpy().astype(np.int64))
    return hist.finalize()


@torch.no_grad()
def code_image(model, hist, img, setting_id, model_id):
    """1 枚の画像を符号化・復号し，再構成とビット数を返す．

    Args:
        model (CAE): モデル．
        hist (ec.HistogramModel): 確率モデル．
        img (np.ndarray): 形状 (H, W, 3) の uint8 画像．
        setting_id (int): レート設定番号．
        model_id (int): アンサンブル内のモデル番号．
    Returns:
        dict: 再構成画像 "rec"，ビットストリーム "stream" と各種ビット数．
    """
    h, w = img.shape[:2]
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda()
    z = model.encode(x)
    z_np = z[0].cpu().numpy().astype(np.int64)
    stream, payload = ec.encode(z_np, hist, h, w, model_id, setting_id)
    z_dec, _ = ec.decode(stream, hist)
    if not np.array_equal(z_dec, z_np):
        raise RuntimeError("復号した係数が符号化前と一致しない")
    rec = model.decode(torch.from_numpy(z_dec).float()[None].cuda())
    rec = rec[0].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
    pixels = h * w
    return {"rec": rec, "stream": stream,
            "bpp": (payload + ENSEMBLE_BYTES) * 8 / pixels, "bpp_payload": payload * 8 / pixels,
            "bpp_file": len(stream) * 8 / pixels,
            "bpp_est_gsm": -model.gsm.discrete_log2_prob(z).sum().item() / pixels,
            "bpp_est_hist": hist.ideal_bits(z_np) / pixels,
            "nonzero_ratio": float((z_np != 0).mean())}


def _metrics_job(args):
    """並列計算用: 原画像と再構成の評価指標を計算する．

    Args:
        args (tuple): (原画像, 再構成画像)．
    Returns:
        dict: 評価指標．
    """
    return all_metrics(*args)


def evaluate_model(alpha, channels, seed=0, model_id=0, save_images=True, pool=None):
    """1 モデルの全レート設定を Kodak 24 枚で評価する．

    Args:
        alpha (float): 学習時の α．
        channels (int): 潜在チャネル数．
        seed (int): seed．
        model_id (int): アンサンブル内のモデル番号（ビットストリームに記録）．
        save_images (bool): 再構成画像とビットストリームを保存するか．
        pool (multiprocessing.Pool | None): 評価指標計算用のプロセスプール．
    Returns:
        pd.DataFrame: 画像別の評価結果．
    """
    directory = model_dir(alpha, channels, seed)
    out_dir = os.path.join(directory, "kodak")
    os.makedirs(out_dir, exist_ok=True)
    model, weights = load_trained(alpha, channels, seed)
    kodak = load_kodak()
    rows, done = _load_previous(out_dir)
    for setting_id, (name, target, log_scales) in enumerate(rate_settings(directory)):
        if name in done:
            continue
        model.log_scales.data = torch.zeros(channels, device="cuda") if log_scales is None else \
            torch.tensor(log_scales, device="cuda")
        hist = build_histogram(model)
        with open(os.path.join(out_dir, f"histogram_{name}.json"), "w") as f:
            json.dump(hist.state_dict(), f)
        coded = [code_image(model, hist, img, setting_id, model_id) for _, img in tqdm(kodak, desc=name, leave=False)]
        metrics = pool.map(_metrics_job, [(img, c["rec"]) for (_, img), c in zip(kodak, coded)]) if pool else \
            [_metrics_job((img, c["rec"])) for (_, img), c in zip(kodak, coded)]
        for (img_name, _), c, m in zip(kodak, coded, metrics):
            if save_images:
                _save_outputs(out_dir, name, img_name, c)
            rows.append({"model": f"{alpha}_{channels}", "train_alpha": alpha, "channels": channels, "setting": name,
                         "setting_id": setting_id, "target_alpha": target if target is not None else alpha,
                         "image": img_name, **{k: v for k, v in c.items() if k not in ("rec", "stream")}, **m,
                         "weights": os.path.basename(weights)})
        df = pd.DataFrame(rows)
        tqdm.write(f"[{alpha}_{channels}] {name}: bpp {df[df.setting == name].bpp.mean():.3f} "
                   f"psnr {df[df.setting == name].psnr.mean():.2f} ssim {df[df.setting == name].ssim.mean():.4f} "
                   f"msssim {df[df.setting == name].msssim.mean():.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
    summary = df.groupby(["model", "setting", "setting_id", "target_alpha"], as_index=False).mean(numeric_only=True)
    summary.sort_values("bpp").to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return df


def _load_previous(out_dir):
    """既存の評価結果を読み込む（評価済みの設定は再評価しない）．

    Args:
        out_dir (str): 出力ディレクトリ．
    Returns:
        tuple[list[dict], set[str]]: (既存の行, 評価済みの設定名の集合)．
    """
    path = os.path.join(out_dir, "per_image.csv")
    if not os.path.exists(path):
        return [], set()
    prev = pd.read_csv(path)
    prev = prev[prev.groupby("setting").image.transform("count") == 24]
    return prev.to_dict("records"), set(prev.setting)


def _save_outputs(out_dir, setting, img_name, coded):
    """ビットストリームと再構成画像を保存する．

    Args:
        out_dir (str): 出力ディレクトリ．
        setting (str): レート設定名．
        img_name (str): 画像名．
        coded (dict): code_image の戻り値．
    Returns:
        None
    """
    for sub in ["bitstreams", "reconstructions"]:
        os.makedirs(os.path.join(out_dir, sub, setting), exist_ok=True)
    with open(os.path.join(out_dir, "bitstreams", setting, f"{img_name}.bin"), "wb") as f:
        f.write(coded["stream"])
    Image.fromarray(coded["rec"]).save(os.path.join(out_dir, "reconstructions", setting, f"{img_name}.png"))


def main():
    """全モデルを評価する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="例: 0.2_64")
    parser.add_argument("--no-images", action="store_true", help="再構成画像・ビットストリームを保存しない")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with mp.Pool(12) as pool:
        for model_id, (alpha, ch) in enumerate(MODEL_CONFIGS):
            if args.only not in (None, f"{alpha}_{ch}"):
                continue
            if not os.path.exists(model_dir(alpha, ch)):
                print("未学習のためスキップ:", alpha, ch)
                continue
            evaluate_model(alpha, ch, model_id=model_id, save_images=not args.no_images, pool=pool)


if __name__ == "__main__":
    main()
