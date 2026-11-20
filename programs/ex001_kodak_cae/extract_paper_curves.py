"""論文 PDF および著者公開の補足資料 PDF からレート・歪み曲線の数値を抽出する．

対象:
    - reference/theis2017.pdf の Figure 4（Kodak 24 枚平均，JPEG/JP2/Toderici/CAE）
    - reference/theis2017.pdf の Figure 9（13 ページ下段，低・中・高ビットレート CAE 個別の曲線）
    - 著者の補足資料 (https://figshare.com/articles/supplementary_zip/4210152) の
      psnr_by_image.pdf / ssim_by_image.pdf / msssim_by_image.pdf（画像別の曲線）
これらの図は matplotlib のベクタ PDF であり，曲線は折れ線パスとして埋め込まれている．
本スクリプトは折れ線の頂点座標を読み出し，目盛ラベルの位置から求めた線形写像でデータ座標に変換する．
画素からの読み取りではないが，「図から抽出した値」であり，著者が数値表として公開した値ではない点に注意する．

出力: outputs/ex001_kodak_cae/paper_values/{figure4,figure9,by_image}.csv
"""
import os
import re
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import pymupdf

from data import ROOT

PAPER_PDF = os.path.join(ROOT, "reference", "theis2017.pdf")
SUPP_URL = "https://ndownloader.figshare.com/files/6868806"
SUPP_DIR = os.path.join(ROOT, "datasets", "raw", "theis2017_supplementary")
OUT_DIR = os.path.join(ROOT, "outputs", "ex001_kodak_cae", "paper_values")
METRIC_ORDER = ["psnr", "ssim", "msssim"]
FIG4_COLORS = {(1.0, 0.0, 0.0): "JPEG (4:2:0, optimized)", (0.9, 0.7, 0.0): "JPEG 2000",
               (0.4, 0.6, 1.0): "Toderici et al. (2016b)", (0.0, 0.0, 0.0): "CAE (ensemble)"}


def _is_number(text):
    """文字列が数値の目盛ラベルかを判定する．

    Args:
        text (str): 文字列．
    Returns:
        bool: 数値なら True．
    """
    return re.fullmatch(r"-?\d+(\.\d+)?", text) is not None


def _find_panels(page):
    """x 軸の目盛ラベル "0.0"〜"3.0" の並びからパネル（axes）を検出し，座標変換を求める．

    Args:
        page (pymupdf.Page): PDF のページ．
    Returns:
        list[dict]: 各パネルの {"x0","x1","top","bottom","ax","bx","ay","by"}．
            データ座標は x = ax * X + bx，y = ay * Y + by（X, Y は PDF 座標）．
    """
    words = [w for w in page.get_text("words") if _is_number(w[4])]
    xlabels = [w for w in words if w[4] in {"0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"}]
    panels = []
    for w0 in [w for w in xlabels if w[4] == "0.0"]:
        row = sorted([w for w in xlabels if abs(w[1] - w0[1]) < 1.0 and 0 <= w[0] - w0[0] < 300], key=lambda w: w[0])
        seq = [w[4] for w in row]
        if seq[:7] != ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]:
            continue
        row = row[:7]
        xc = np.array([(w[0] + w[2]) / 2 for w in row])
        ax, bx = np.polyfit(xc, [float(w[4]) for w in row], 1)
        span = xc[-1] - xc[0]
        ylabels = [w for w in words if -5 < w0[0] - w[2] < 0.15 * span and w[1] < w0[1] - 1 and w0[1] - w[1] < 1.2 * span]
        ylabels = _contiguous_column(ylabels)
        yc = np.array([(w[1] + w[3]) / 2 for w in ylabels])
        ay, by = np.polyfit(yc, [float(w[4]) for w in ylabels], 1)
        panels.append({"x0": xc[0] - 8, "x1": xc[-1] + 30, "top": yc.min() - 10, "bottom": w0[1],
                       "ax": ax, "bx": bx, "ay": ay, "by": by, "ymin": min(float(w[4]) for w in ylabels)})
    return panels


def _contiguous_column(words):
    """y 軸の目盛ラベル候補から，右端が揃った 1 列を選ぶ．

    Args:
        words (list): pymupdf の words 要素のリスト．
    Returns:
        list: 同じ列に属するラベル．
    """
    if not words:
        return words
    right = max(w[2] for w in words)
    return [w for w in words if abs(w[2] - right) < 1.5]


def _curves(page, allow_gray=False):
    """ページ内の折れ線（2 線分以上）を (色, 頂点配列) として返す．

    Args:
        page (pymupdf.Page): PDF のページ．
        allow_gray (bool): 灰色の曲線も含めるか（Figure 9 用）．
    Returns:
        list[tuple[tuple, np.ndarray]]: (RGB 色, 形状 (n, 2) の PDF 座標) のリスト．
    """
    out = []
    for d in page.get_drawings():
        if not d["color"] or len(d["items"]) < 2 or any(it[0] != "l" for it in d["items"]):
            continue
        color = tuple(round(c, 2) for c in d["color"])
        if not allow_gray and len(set(color)) == 1 and color[0] > 0.1:  # 灰色の補助線を除く
            continue
        pts = [d["items"][0][1]] + [it[2] for it in d["items"]]
        out.append((color, np.array([[p.x, p.y] for p in pts])))
    return out


def _assign(panels, pts):
    """折れ線の頂点を最も多く含むパネルの番号を返す（軸範囲外の頂点があっても正しく割り当てる）．

    Args:
        panels (list[dict]): _find_panels の戻り値．
        pts (np.ndarray): 形状 (n, 2) の PDF 座標．
    Returns:
        int | None: パネル番号（該当なしは None）．
    """
    counts = [int(((pts[:, 0] >= p["x0"]) & (pts[:, 0] <= p["x1"]) &
                   (pts[:, 1] >= p["top"] - 5) & (pts[:, 1] <= p["bottom"])).sum()) for p in panels]
    if not counts or max(counts) == 0:
        return None
    return int(np.argmax(counts))


def _to_data(panel, pts):
    """PDF 座標をデータ座標に変換する．

    Args:
        panel (dict): パネル情報．
        pts (np.ndarray): 形状 (n, 2) の PDF 座標．
    Returns:
        np.ndarray: 形状 (n, 2) の (bpp, 指標値)．
    """
    return np.stack([panel["ax"] * pts[:, 0] + panel["bx"], panel["ay"] * pts[:, 1] + panel["by"]], 1)


def extract_figure(page_index, labels, allow_gray=False, min_bottom=0.0):
    """論文の 1 行 3 パネル（PSNR, SSIM, MS-SSIM）の図から曲線を抽出する．

    Args:
        page_index (int): 0 始まりのページ番号．
        labels (dict): 色 -> 手法名の対応．
        allow_gray (bool): 灰色の曲線も対象にするか．
        min_bottom (float): この y 座標より下にあるパネルのみ対象にする（同一ページの別の図を除く）．
    Returns:
        pd.DataFrame: 列 method, metric, bpp, value．
    """
    page = pymupdf.open(PAPER_PDF)[page_index]
    panels = sorted([p for p in _find_panels(page) if p["bottom"] > min_bottom], key=lambda p: p["x0"])
    rows = []
    for color, pts in _curves(page, allow_gray):
        if pts[:, 1].max() < min_bottom - 150:
            continue
        i = _assign(panels, pts)
        if i is None or color not in labels:
            continue
        for bpp, val in _to_data(panels[i], pts):
            rows.append({"method": labels[color], "metric": METRIC_ORDER[i], "bpp": bpp, "value": val})
    return pd.DataFrame(rows)


def _figure9_labels():
    """Figure 9（13 ページ下段）の曲線の色と α の対応を返す．

    凡例（左から α=0.01 が灰色 0.7，α=0.05 が灰色 0.5，α=0.2 が黒）の色に基づく．

    Returns:
        dict: 色 -> "CAE (alpha=...)"．
    """
    return {(0.7, 0.7, 0.7): "CAE (alpha=0.01)", (0.5, 0.5, 0.5): "CAE (alpha=0.05)", (0.0, 0.0, 0.0): "CAE (alpha=0.2)"}


def download_supplementary():
    """著者の補足資料をダウンロードして展開する（既にあればスキップ）．

    Returns:
        str: 補足資料ディレクトリのパス．
    """
    target = os.path.join(SUPP_DIR, "supplementary")
    if os.path.isdir(target):
        return target
    os.makedirs(SUPP_DIR, exist_ok=True)
    archive = os.path.join(SUPP_DIR, "supplementary.tar.gz")
    urllib.request.urlretrieve(SUPP_URL, archive)
    with tarfile.open(archive) as tf:
        tf.extractall(SUPP_DIR, filter="data")
    return target


def extract_by_image(supp_dir):
    """補足資料の画像別 PDF から 24 枚それぞれの曲線を抽出する．

    Args:
        supp_dir (str): 補足資料ディレクトリ．
    Returns:
        pd.DataFrame: 列 image, method, metric, bpp, value．
    """
    rows = []
    for metric in METRIC_ORDER:
        page = pymupdf.open(os.path.join(supp_dir, f"{metric}_by_image.pdf"))[0]
        panels = _find_panels(page)
        titles = [w for w in page.get_text("words") if w[4].startswith("kodim")]
        for p in panels:
            title = min((t for t in titles if t[3] <= p["top"] + 10), key=lambda t: (p["top"] - t[3]) ** 2 + ((t[0] + t[2]) / 2 - (p["x0"] + p["x1"]) / 2) ** 2)
            p["image"] = title[4].replace(".png", "")
        for color, pts in _curves(page):
            i = _assign(panels, pts)
            if i is None or color not in FIG4_COLORS:
                continue
            for bpp, val in _to_data(panels[i], pts):
                rows.append({"image": panels[i]["image"], "method": FIG4_COLORS[color], "metric": metric, "bpp": bpp, "value": val})
    return pd.DataFrame(rows)


def main():
    """全ての図から値を抽出し CSV に保存する．

    Returns:
        None
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    fig4 = extract_figure(6, FIG4_COLORS)
    fig4.to_csv(os.path.join(OUT_DIR, "figure4.csv"), index=False)
    fig9 = extract_figure(12, _figure9_labels(), allow_gray=True, min_bottom=400.0)
    fig9.to_csv(os.path.join(OUT_DIR, "figure9.csv"), index=False)
    by_image = extract_by_image(download_supplementary())
    by_image.to_csv(os.path.join(OUT_DIR, "by_image.csv"), index=False)
    for name, df in [("figure4", fig4), ("figure9", fig9), ("by_image", by_image)]:
        print(name, df.groupby(["method", "metric"]).size().to_dict() if len(df) else "empty")


if __name__ == "__main__":
    main()
