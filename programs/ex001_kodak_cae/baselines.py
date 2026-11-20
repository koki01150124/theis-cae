"""JPEG（4:2:0，optimized）と JPEG 2000 の Kodak 24 枚に対するレート・歪み評価．

論文 3.3 節，Appendix A.2 に合わせ，ヘッダのビットは数えない．ただし optimized JPEG では
Huffman テーブル (DHT) のビットは数える．具体的には次のように数える．
    - JPEG    : DHT セグメント + エントロピー符号化データ（SOS ヘッダ以降〜EOI 直前）
    - JPEG2000: 生のコードストリーム (.j2k) のうち，SOD マーカ以降のタイルデータ（EOC を除く）
ファイル全体のサイズによる bpp も bpp_file として併記する．
符号化器: JPEG は Pillow (libjpeg-turbo)，JPEG 2000 は Pillow (OpenJPEG, 9/7 非可逆ウェーブレット)．
論文で用いた符号化器の実装は明記されていないため，これらは代替の実装である．

出力: outputs/ex001_kodak_cae/baselines/{JPEG,JPEG2000}/per_image.csv
"""
import io
import multiprocessing as mp
import os
import struct

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from data import ROOT, load_kodak
from metrics import all_metrics, ssim_gray

OUT_DIR = os.path.join(ROOT, "outputs", "ex001_kodak_cae", "baselines")
JPEG_QUALITIES = list(range(1, 101))
JP2_TARGET_BPP = [round(v, 3) for v in np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 3.01, 0.1)])]


def jpeg_counted_bytes(data):
    """JPEG ファイルのうち DHT とエントロピー符号化データのバイト数を数える．

    Args:
        data (bytes): JPEG ファイルの内容．
    Returns:
        int: 数えたバイト数．
    """
    pos, counted = 2, 0
    while pos < len(data):
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if marker == 0xC4:  # DHT
            counted += length + 2
        if marker == 0xDA:  # SOS: ヘッダの後ろにエントロピー符号化データが続く
            scan_start = pos + 2 + length
            scan_end = data.rfind(b"\xff\xd9")
            counted += scan_end - scan_start
            break
        pos += 2 + length
    return counted


def j2k_counted_bytes(data):
    """JPEG 2000 コードストリームのうち SOD 以降のタイルデータのバイト数を数える．

    Args:
        data (bytes): .j2k コードストリーム．
    Returns:
        int: 数えたバイト数（全タイルパートの合計）．
    """
    counted, pos = 0, data.find(b"\xff\x90")  # 最初の SOT
    while pos >= 0 and pos < len(data) - 2:
        psot = struct.unpack(">I", data[pos + 6:pos + 10])[0]
        tile_end = pos + psot if psot > 0 else len(data) - 2
        sod = data.find(b"\xff\x93", pos)
        counted += tile_end - (sod + 2)
        pos = tile_end if data[tile_end:tile_end + 2] == b"\xff\x90" else -1
    return counted


def _jpeg(img, quality):
    """JPEG (4:2:0, optimized) で符号化・復号する．

    Args:
        img (np.ndarray): 形状 (H, W, 3) の uint8 画像．
        quality (int): 品質係数．
    Returns:
        tuple[np.ndarray, int, int]: (復号画像, 数えたバイト数, ファイルのバイト数)．
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=2, optimize=True)
    data = buf.getvalue()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), jpeg_counted_bytes(data), len(data)


def _jp2(img, target_bpp):
    """JPEG 2000 で目標 bpp（ファイル全体）を指定して符号化・復号する．

    Args:
        img (np.ndarray): 形状 (H, W, 3) の uint8 画像．
        target_bpp (float): 目標ビットレート．
    Returns:
        tuple[np.ndarray, int, int]: (復号画像, 数えたバイト数, ファイルのバイト数)．
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG2000", no_jp2=True, irreversible=True,
                              quality_mode="rates", quality_layers=[24.0 / target_bpp], mct=1)
    data = buf.getvalue()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), j2k_counted_bytes(data), len(data)


def _evaluate_image(args):
    """1 枚の画像について全設定の符号化・評価を行う．

    Args:
        args (tuple): (方式名, 画像名, 画像配列)．
    Returns:
        list[dict]: 設定ごとの評価結果．
    """
    method, name, img = args
    settings = JPEG_QUALITIES if method == "JPEG" else JP2_TARGET_BPP
    codec = _jpeg if method == "JPEG" else _jp2
    pixels = img.shape[0] * img.shape[1]
    rows = []
    for s in settings:
        rec, counted, total = codec(img, s)
        row = {"image": name, "setting": s, "bpp": counted * 8 / pixels, "bpp_file": total * 8 / pixels}
        row.update(all_metrics(img, rec))
        row["ssim_gray"] = ssim_gray(img, rec)
        rows.append(row)
    return rows


def main(processes=12):
    """JPEG と JPEG 2000 の評価を並列に実行し CSV に保存する．

    Args:
        processes (int): 並列プロセス数．
    Returns:
        None
    """
    images = load_kodak()
    for method in ["JPEG", "JPEG2000"]:
        out = os.path.join(OUT_DIR, method)
        os.makedirs(out, exist_ok=True)
        with mp.Pool(processes) as pool:
            results = list(tqdm(pool.imap(_evaluate_image, [(method, n, im) for n, im in images]),
                                total=len(images), desc=method, leave=False))
        df = pd.DataFrame([r for rows in results for r in rows])
        df.to_csv(os.path.join(out, "per_image.csv"), index=False)
        print(method, len(df), "rows")


if __name__ == "__main__":
    main()
