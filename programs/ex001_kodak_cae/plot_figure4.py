"""論文 Figure 4 形式のレート・歪み曲線（PSNR / SSIM / MS-SSIM）を作成する．

平均化の方法:
    画像ごとにレート・歪み曲線を bpp の格子点で線形補間し，24 枚すべてで値が定義される格子点について
    平均する．著者公開の画像別曲線をこの方法で平均すると論文 Figure 4 と一致することを確認している．
CAE (ensemble):
    論文 3.3 節に従い，画像ごと・ビットレートごとに歪み (MSE) が最小のモデルを選ぶ．
    各モデルの曲線を同じ bpp で補間し，MSE 最小のモデルの PSNR / SSIM / MS-SSIM を採用する．
    bpp にはモデル選択情報の 1 byte を含む（evaluate.py の bpp 列）．

出力: outputs/ex001_kodak_cae/figures/
    figure4.{png,pdf}            : 実測 bpp による Figure 4 形式の図
    figure4_estimated.{png,pdf}  : 推定 bpp (GSM) による参考図（実測と区別）
    figure9.{png,pdf}            : 低・中・高ビットレートの各モデルの曲線（論文 Figure 9 形式）
    curves.csv                   : 図に描いた平均曲線の数値
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import ROOT  # noqa: E402

EXP_DIR = os.path.join(ROOT, "outputs", "ex001_kodak_cae")
FIG_DIR = os.path.join(EXP_DIR, "figures")
GRID = np.round(np.arange(0.05, 3.001, 0.025), 3)
METRICS = [("psnr", "PSNR [dB]", (20, 45)), ("ssim", "SSIM", (0.65, 1.0)), ("msssim", "MS-SSIM", (0.75, 1.0))]
STYLE = {  # 論文 Figure 4 の配色を踏襲し，再現値は実線，論文から抽出した値は破線で区別する
    "JPEG (4:2:0, optimized)": "#ff0000",
    "JPEG 2000": "#e6b300",
    "Toderici et al. (2016b)": "#6699ff",
    "CAE (ensemble)": "#000000",
}
# 表示対象（変数で選択可能）: (曲線名, 色のキー, 線種, 線幅)
SHOW = [
    ("JPEG (4:2:0, optimized) [ours]", "JPEG (4:2:0, optimized)", "-", 1.3),
    ("JPEG (4:2:0, optimized) [paper]", "JPEG (4:2:0, optimized)", "--", 1.0),
    ("JPEG 2000 [ours]", "JPEG 2000", "-", 1.3),
    ("JPEG 2000 [paper]", "JPEG 2000", "--", 1.0),
    ("Toderici et al. (2016b) [paper]", "Toderici et al. (2016b)", "--", 1.0),
    ("CAE (ensemble) [paper]", "CAE (ensemble)", "--", 1.0),
    ("CAE (ensemble) [ours]", "CAE (ensemble)", "-", 2.0),
]


def mean_curve(per_image, bpp_col="bpp", metrics=("psnr", "ssim", "msssim")):
    """画像ごとに bpp 格子で補間し，全画像で定義される格子点の平均を返す．

    Args:
        per_image (pd.DataFrame): 列 image, bpp_col, 各指標を持つ画像別の結果．
        bpp_col (str): 横軸に用いる bpp の列名．
        metrics (tuple[str]): 平均する指標名．
    Returns:
        pd.DataFrame: 列 bpp と各指標の平均曲線．
    """
    stacks = {m: [] for m in metrics}
    for _, g in per_image.groupby("image"):
        g = g.sort_values(bpp_col)
        for m in metrics:
            stacks[m].append(np.interp(GRID, g[bpp_col], g[m], left=np.nan, right=np.nan))
    out = pd.DataFrame({"bpp": GRID})
    for m in metrics:
        out[m] = np.mean(np.array(stacks[m]), axis=0)
    return out.dropna()


def ensemble_per_image(cae, bpp_col="bpp"):
    """画像・ビットレートごとに MSE 最小のモデルを選んだアンサンブルの画像別曲線を作る．

    Args:
        cae (pd.DataFrame): evaluate.py の画像別結果を全モデル分連結したもの．
        bpp_col (str): 横軸に用いる bpp の列名．
    Returns:
        pd.DataFrame: 列 image, bpp_col, psnr, ssim, msssim, mse, chosen_model．
    """
    rows = []
    for img, g in cae.groupby("image"):
        curves = {}
        for model, gm in g.groupby("model"):
            gm = gm.sort_values(bpp_col)
            curves[model] = {k: np.interp(GRID, gm[bpp_col], gm[k], left=np.nan, right=np.nan)
                             for k in ["mse", "psnr", "ssim", "msssim"]}
        models = list(curves)
        mse = np.stack([curves[m]["mse"] for m in models])
        valid = ~np.all(np.isnan(mse), axis=0)
        best = np.nanargmin(np.where(np.isnan(mse), np.inf, mse), axis=0)
        for i in np.where(valid)[0]:
            c = curves[models[best[i]]]
            rows.append({"image": img, bpp_col: GRID[i], "psnr": c["psnr"][i], "ssim": c["ssim"][i],
                         "msssim": c["msssim"][i], "mse": c["mse"][i], "chosen_model": models[best[i]]})
    return pd.DataFrame(rows)


def load_cae_results():
    """evaluate.py が出力した全モデルの画像別結果を読み込む．

    Returns:
        pd.DataFrame | None: 連結した結果（存在しない場合 None）．
    """
    paths = sorted(glob.glob(os.path.join(EXP_DIR, "CAE", "*", "*", "kodak", "per_image.csv")))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["mse"] = 255.0 ** 2 / 10 ** (df["psnr"] / 10)
    return df


def paper_curves(path, method):
    """論文の図から抽出した曲線を指標ごとに返す．

    Args:
        path (str): 抽出値の CSV．
        method (str): 手法名．
    Returns:
        dict[str, pd.DataFrame]: 指標名 -> 列 bpp, value．
    """
    df = pd.read_csv(path)
    return {m: df[(df.method == method) & (df.metric == m)].sort_values("bpp") for m, _, _ in METRICS}


def build_curves(bpp_col="bpp"):
    """図に描く全曲線（指標ごとの (bpp, 値)）を作る．

    Args:
        bpp_col (str): CAE の横軸に用いる bpp 列（"bpp" または "bpp_est_gsm"）．
    Returns:
        dict[str, dict[str, pd.DataFrame]]: 曲線名 -> 指標名 -> 列 bpp, value．
    """
    curves = {}
    fig4 = os.path.join(EXP_DIR, "paper_values", "figure4.csv")
    for method in STYLE:
        curves[f"{method} [paper]"] = paper_curves(fig4, method)
    for name, method in [("JPEG", "JPEG (4:2:0, optimized)"), ("JPEG2000", "JPEG 2000")]:
        path = os.path.join(EXP_DIR, "baselines", name, "per_image.csv")
        if os.path.exists(path):
            mc = mean_curve(pd.read_csv(path))
            curves[f"{method} [ours]"] = {m: mc.rename(columns={m: "value"})[["bpp", "value"]] for m, _, _ in METRICS}
    cae = load_cae_results()
    if cae is not None:
        ens = ensemble_per_image(cae, bpp_col)
        mc = mean_curve(ens, bpp_col).rename(columns={bpp_col: "bpp"})
        curves["CAE (ensemble) [ours]"] = {m: mc.rename(columns={m: "value"})[["bpp", "value"]] for m, _, _ in METRICS}
        ens.to_csv(os.path.join(FIG_DIR, f"ensemble_per_image_{bpp_col}.csv"), index=False)
    return curves


def draw(curves, show, out_name, title_suffix=""):
    """3 つの指標を横並びにした図を描いて PNG と PDF で保存する．

    Args:
        curves (dict): build_curves の戻り値．
        show (list[tuple]): (曲線名, 色のキー, 線種, 線幅) のリスト．
        out_name (str): 出力ファイル名（拡張子なし）．
        title_suffix (str): 図の上部に付ける注記．
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (m, label, ylim) in zip(axes, METRICS):
        for name, color_key, ls, lw in show:
            if name not in curves:
                continue
            c = curves[name][m]
            ax.plot(c["bpp"], c["value"], ls, color=STYLE.get(color_key, color_key), lw=lw, label=name)
        ax.set_xlim(0, 3)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Bit rate [bpp]")
        ax.set_ylabel(label)
        ax.grid(color="#dddddd", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    if title_suffix:
        fig.text(0.5, 0.005, title_suffix, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{out_name}.{ext}"), dpi=200)
    plt.close(fig)


def draw_figure9():
    """各モデル（低・中・高ビットレート）の平均曲線を論文 Figure 9 と比較する図を描く．

    Returns:
        pd.DataFrame | None: 各モデル・設定の平均値（設定ごとの 24 枚平均）．
    """
    cae = load_cae_results()
    if cae is None:
        return None
    fig9 = os.path.join(EXP_DIR, "paper_values", "figure9.csv")
    colors = {"0.01_96": "#bbbbbb", "0.05_96": "#777777", "0.2_64": "#000000"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (m, label, ylim) in zip(axes, METRICS):
        for model, g in cae.groupby("model"):
            mc = mean_curve(g)
            ax.plot(mc["bpp"], mc[m], "-", color=colors.get(model, "C0"), lw=1.8,
                    label=f"CAE (alpha={model.split('_')[0]}) [ours]")
            pts = g.groupby("setting").mean(numeric_only=True)
            ax.plot(pts["bpp"], pts[m], "o", color=colors.get(model, "C0"), ms=3)
        for model, color in [("CAE (alpha=0.01)", "#bbbbbb"), ("CAE (alpha=0.05)", "#777777"), ("CAE (alpha=0.2)", "#000000")]:
            c = paper_curves(fig9, model)[m]
            ax.plot(c["bpp"], c["value"], "--", color=color, lw=1.0, label=f"{model} [paper]")
        ax.set_xlim(0, 3)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Bit rate [bpp]")
        ax.set_ylabel(label)
        ax.grid(color="#dddddd", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.005, "solid: this reproduction (lines: per-image interpolation then mean; markers: per-setting means) / "
             "dashed: values extracted from Figure 9 of Theis et al. (2017)",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"figure9.{ext}"), dpi=200)
    plt.close(fig)
    return cae.groupby(["model", "setting"], as_index=False).mean(numeric_only=True)


def save_curves(curves, path):
    """描画した曲線を縦持ちの CSV に保存する．

    Args:
        curves (dict): build_curves の戻り値．
        path (str): 保存先．
    Returns:
        None
    """
    rows = []
    for name, per_metric in curves.items():
        for m, c in per_metric.items():
            rows += [{"curve": name, "metric": m, "bpp": b, "value": v} for b, v in zip(c["bpp"], c["value"])]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    """全ての図と曲線の CSV を作成する．

    Returns:
        None
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    curves = build_curves("bpp")
    draw(curves, SHOW, "figure4", "solid: this reproduction (measured bpp, range coder) / "
         "dashed: values extracted from the vector figures of Theis et al. (2017)")
    save_curves(curves, os.path.join(FIG_DIR, "curves.csv"))
    if load_cae_results() is not None:
        est = build_curves("bpp_est_gsm")
        est["CAE (ensemble, estimated bpp) [ours]"] = est.pop("CAE (ensemble) [ours]")
        show = [s for s in SHOW if not s[0].startswith("CAE (ensemble) [ours]")] + \
            [("CAE (ensemble, estimated bpp) [ours]", "#000000", ":", 2.0)]
        draw(est, show, "figure4_estimated", "REFERENCE ONLY: CAE plotted against GSM-estimated bpp "
             "(not measured bitstream size)")
        save_curves(est, os.path.join(FIG_DIR, "curves_estimated.csv"))
        summary = draw_figure9()
        summary.to_csv(os.path.join(FIG_DIR, "per_setting_summary.csv"), index=False)
    print("saved:", FIG_DIR)


if __name__ == "__main__":
    main()
