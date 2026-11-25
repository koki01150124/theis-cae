"""発表スライド用の図（文字を大きくした版）を作成する．

出力: outputs/ex001_kodak_cae/figures/slides/
    rd_curves.png        : Figure 4 形式（PSNR / SSIM / MS-SSIM）
    model_curves.png     : 各モデルの曲線と論文 Figure 9 の比較（PSNR / SSIM）
    training.png         : 検証 PSNR と有効チャネル数の推移
    recon_*.png          : 復元画像の切り出し（原画像，CAE，JPEG，JPEG 2000）
    recon_info.json      : 復元画像の bpp と PSNR
"""
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import KODAK_DIR  # noqa: E402
from metrics import psnr  # noqa: E402
from plot_figure4 import EXP_DIR, mean_curve, paper_curves  # noqa: E402

OUT = os.path.join(EXP_DIR, "figures", "slides")
# スライド上で実寸（約 33 cm 幅）になるように図の大きさを決め，文字を 20pt 程度にする
plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "legend.fontsize": 20,
                     "xtick.labelsize": 18, "ytick.labelsize": 18})
COLORS = {"CAE": "#000000", "JP2": "#e6b300", "JPEG": "#ff0000", "TOD": "#6699ff"}
PANELS = {"psnr": ("PSNR [dB]", (20, 45)), "ssim": ("SSIM", (0.65, 1.0)), "msssim": ("MS-SSIM", (0.75, 1.0))}


def _style(ax, metric):
    """軸の体裁を整える．

    Args:
        ax (matplotlib.axes.Axes): 軸．
        metric (str): 指標名．
    Returns:
        None
    """
    label, ylim = PANELS[metric]
    ax.set_xlim(0, 3)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Bit rate [bpp]")
    ax.set_ylabel(label)
    ax.grid(color="#dddddd", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def rd_curves():
    """Figure 4 形式の 3 指標の図を描く．

    Returns:
        None
    """
    c = pd.read_csv(os.path.join(EXP_DIR, "figures", "curves.csv"))
    show = [("CAE (ensemble) [ours]", "CAE", "-", 3.0, "CAE（再現）"),
            ("CAE (ensemble) [paper]", "CAE", "--", 1.8, "CAE（論文）"),
            ("JPEG 2000 [ours]", "JP2", "-", 2.0, "JPEG 2000（本評価）"),
            ("JPEG 2000 [paper]", "JP2", "--", 1.5, "JPEG 2000（論文）"),
            ("JPEG (4:2:0, optimized) [ours]", "JPEG", "-", 2.0, "JPEG（本評価）"),
            ("Toderici et al. (2016b) [paper]", "TOD", "--", 1.5, "Toderici et al.（論文）")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0))
    for ax, metric in zip(axes, PANELS):
        for name, col, ls, lw, label in show:
            d = c[(c.curve == name) & (c.metric == metric)]
            ax.plot(d.bpp, d.value, ls, color=COLORS[col], lw=lw, label=label)
        _style(ax, metric)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=0.6)
    fig.savefig(os.path.join(OUT, "rd_curves.png"), dpi=200)
    plt.close(fig)


def model_curves():
    """各モデルの曲線と論文 Figure 9 を比較する図を描く．

    Returns:
        None
    """
    fig9 = os.path.join(EXP_DIR, "paper_values", "figure9.csv")
    models = [("0.01_96", "0.01", "#b0b0b0"), ("0.05_96", "0.05", "#707070"), ("0.2_64", "0.2", "#000000")]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax, metric in zip(axes, ["psnr", "ssim"]):
        for model, a, color in models:
            d = pd.read_csv(os.path.join(EXP_DIR, "CAE", model, "0", "kodak", "per_image.csv"))
            pts = d.groupby("setting").mean(numeric_only=True).sort_values("bpp")
            ax.plot(pts.bpp, pts[metric], "o-", color=color, lw=2.2, ms=5, label=f"α={a}（再現）")
            p = paper_curves(fig9, f"CAE (alpha={a})")[metric]
            ax.plot(p.bpp, p.value, "--", color=color, lw=2.0, label=f"α={a}（論文）")
            base = pts.loc["base"]
            ax.plot(base.bpp, base[metric], "*", color="#d62728", ms=16, zorder=5)
        _style(ax, metric)
    axes[0].set_ylim(22, 45)
    axes[1].set_ylim(0.70, 1.0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(os.path.join(OUT, "model_curves.png"), dpi=200)
    plt.close(fig)


def training():
    """検証 PSNR と有効チャネル数の推移を描く．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for model, label, color in [("0.01_96", "α=0.01, 96ch", "#1f77b4"), ("0.05_96", "α=0.05, 96ch", "#ff7f0e"),
                                ("0.2_64", "α=0.2, 64ch", "#2ca02c")]:
        with open(os.path.join(EXP_DIR, "CAE", model, "0", "log.json")) as f:
            log = pd.DataFrame(json.load(f)).iloc[1:]
        with open(os.path.join(EXP_DIR, "CAE", model, "0", "finished.json")) as f:
            drop = json.load(f)["lr_reduced_at"]
        axes[0].plot(log.global_step / 1e4, log.val_psnr, color=color, lw=2.2, label=label)
        axes[1].plot(log.global_step / 1e4, log.enabled, color=color, lw=2.2, label=label)
        for ax in axes:
            ax.axvline(drop / 1e4, color=color, ls=":", lw=1.6)
    axes[0].set_ylabel("検証 PSNR [dB]")
    axes[1].set_ylabel("有効なチャネル数")
    for ax in axes:
        ax.set_xlabel("更新回数 [×10⁴]")
        ax.grid(color="#dddddd", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[1].legend(loc="lower right", frameon=False, fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training.png"), dpi=200)
    plt.close(fig)


def _jpeg_at_least(img, bpp_target):
    """bpp が目標以上となる最小の品質係数で JPEG 符号化した画像と bpp を返す．

    Args:
        img (np.ndarray): 形状 (H, W, 3) の uint8 画像．
        bpp_target (float): 目標 bpp．
    Returns:
        tuple[np.ndarray, float]: (復号画像, bpp)．
    """
    from baselines import _jpeg
    pixels = img.shape[0] * img.shape[1]
    for q in range(1, 101):
        rec, counted, _ = _jpeg(img, q)
        if counted * 8 / pixels >= bpp_target:
            return rec, counted * 8 / pixels
    return rec, counted * 8 / pixels


def reconstructions(image="kodim04", target=0.375, box=(150, 190, 470, 430)):
    """論文 Figure 5 と同様の方法で同程度の bpp の復元画像を切り出す．

    CAE は目標 bpp を超えない最大の設定（全モデルの全設定から選択），JPEG は CAE 以上の bpp となる最小の品質係数，
    JPEG 2000 は CAE と同じ bpp を目標に符号化する．

    Args:
        image (str): 画像名．
        target (float): 目標 bpp．
        box (tuple[int]): 切り出し範囲 (left, top, right, bottom)．
    Returns:
        None
    """
    from baselines import _jp2
    orig = np.asarray(Image.open(os.path.join(KODAK_DIR, f"{image}.png")).convert("RGB"))
    rows = []
    for model in ["0.01_96", "0.05_96", "0.2_64"]:
        d = pd.read_csv(os.path.join(EXP_DIR, "CAE", model, "0", "kodak", "per_image.csv"))
        rows.append(d[(d.image == image) & (d.bpp <= target)])
    cand = pd.concat(rows).sort_values("bpp").iloc[-1]
    cae = np.asarray(Image.open(os.path.join(EXP_DIR, "CAE", cand.model, "0", "kodak", "reconstructions",
                                             cand.setting, f"{image}.png")))
    jpg, jpg_bpp = _jpeg_at_least(orig, cand.bpp)
    jp2, counted, _ = _jp2(orig, cand.bpp)
    jp2_bpp = counted * 8 / (orig.shape[0] * orig.shape[1])
    info = {"image": image, "box": box,
            "original": {}, "cae": {"bpp": float(cand.bpp), "psnr": psnr(orig, cae), "model": cand.model, "setting": cand.setting},
            "jpeg": {"bpp": jpg_bpp, "psnr": psnr(orig, jpg)}, "jpeg2000": {"bpp": jp2_bpp, "psnr": psnr(orig, jp2)}}
    for name, img in [("original", orig), ("cae", cae), ("jpeg", jpg), ("jpeg2000", jp2)]:
        Image.fromarray(img).crop(box).save(os.path.join(OUT, f"recon_{name}.png"))
    with open(os.path.join(OUT, "recon_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(info)


def main():
    """全図を作成する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
    rd_curves()
    model_curves()
    training()
    reconstructions()


if __name__ == "__main__":
    main()
