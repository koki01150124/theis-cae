"""量子化係数のエントロピー符号化・復号．

論文 3.1 節に従い，学習データ全体での量子化係数のヒストグラムに Laplace 平滑化（各ビンに +1）を
施した確率を用いてレンジコーダで符号化する．論文が用いた kazuho/rangecoder の代わりに，
同じ range coding 方式の公開実装 constriction (RangeEncoder, 32bit word) を用いる．

ヒストグラムはチャネルごと（空間位置で共有）に作成し，学習データで観測された範囲 [lo, hi] の各整数に
加えて，範囲外の値を表すエスケープ記号を 1 つ設ける．エスケープ後の値は 16bit の一様分布で符号化する．

ビットストリームの構成:
    header : H (uint16), W (uint16), model_id (uint8), setting_id (uint8)  計 6 byte
    payload: レンジコーダの出力（uint32 列）
bpp の集計では論文に合わせ，payload と ensemble 選択情報 1 byte を数え，画像サイズ等のヘッダは
別列 (bpp_file) として記録する．
"""
import struct

import constriction
import numpy as np

ESCAPE_BITS = 16
ESCAPE_OFFSET = 1 << (ESCAPE_BITS - 1)
HEADER_FORMAT = "<HHBB"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)


class HistogramModel:
    """チャネルごとの Laplace 平滑化ヒストグラムによる離散確率モデル．"""

    def __init__(self, num_channels):
        """空のヒストグラムを作成する．

        Args:
            num_channels (int): 係数のチャネル数 K．
        """
        self.num_channels = num_channels
        self.counts = [dict() for _ in range(num_channels)]
        self.lo = None
        self.hi = None
        self.probs = None

    def update(self, z):
        """量子化係数の観測を加える．

        Args:
            z (np.ndarray): 形状 (K, h, w) または (B, K, h, w) の整数配列．
        Returns:
            None
        """
        z = np.asarray(z).astype(np.int64)
        if z.ndim == 4:
            z = z.transpose(1, 0, 2, 3)
        for k in range(self.num_channels):
            values, counts = np.unique(z[k], return_counts=True)
            for v, c in zip(values.tolist(), counts.tolist()):
                self.counts[k][v] = self.counts[k].get(v, 0) + c

    def finalize(self):
        """Laplace 平滑化した確率表を作る（範囲内の各整数とエスケープ記号に +1）．

        Returns:
            HistogramModel: 自身．
        """
        self.lo = np.zeros(self.num_channels, dtype=np.int64)
        self.hi = np.zeros(self.num_channels, dtype=np.int64)
        self.probs = []
        for k in range(self.num_channels):
            keys = list(self.counts[k].keys()) or [0]
            lo, hi = min(keys), max(keys)
            table = np.ones(hi - lo + 2, dtype=np.float64)
            for v, c in self.counts[k].items():
                table[v - lo] += c
            self.lo[k], self.hi[k] = lo, hi
            self.probs.append(table / table.sum())
        return self

    def ideal_bits(self, z):
        """モデルの下での理想符号長（ビット）を計算する（レンジコーダとの照合用）．

        Args:
            z (np.ndarray): 形状 (K, h, w) の整数配列．
        Returns:
            float: 理想符号長 [bit]．
        """
        bits = 0.0
        for k in range(self.num_channels):
            idx, esc = self._symbols(k, z[k].reshape(-1))
            bits += -np.log2(self.probs[k][idx]).sum() + ESCAPE_BITS * len(esc)
        return float(bits)

    def _symbols(self, k, values):
        """値を記号番号に変換し，範囲外の値をエスケープとして分離する．

        Args:
            k (int): チャネル番号．
            values (np.ndarray): 形状 (n,) の整数配列．
        Returns:
            tuple[np.ndarray, np.ndarray]: (形状 (n,) の記号番号, 範囲外の値の配列)．
        """
        values = values.astype(np.int64)
        outside = (values < self.lo[k]) | (values > self.hi[k])
        idx = np.where(outside, self.hi[k] - self.lo[k] + 1, values - self.lo[k]).astype(np.int32)
        return idx, values[outside]

    def state_dict(self):
        """保存用の辞書を返す．

        Returns:
            dict: lo, hi, probs を含む辞書．
        """
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "probs": [p.tolist() for p in self.probs]}

    @classmethod
    def from_state_dict(cls, state):
        """保存した辞書から復元する．

        Args:
            state (dict): state_dict() の戻り値．
        Returns:
            HistogramModel: 復元したモデル．
        """
        model = cls(len(state["lo"]))
        model.lo = np.array(state["lo"], dtype=np.int64)
        model.hi = np.array(state["hi"], dtype=np.int64)
        model.probs = [np.array(p, dtype=np.float64) for p in state["probs"]]
        return model


def encode(z, hist, height, width, model_id=0, setting_id=0):
    """量子化係数をビットストリームに符号化する．

    Args:
        z (np.ndarray): 形状 (K, h, w) の整数配列．
        hist (HistogramModel): 確率モデル．
        height (int): 元画像の高さ．
        width (int): 元画像の幅．
        model_id (int): アンサンブル内のモデル番号．
        setting_id (int): レート設定の番号．
    Returns:
        tuple[bytes, int]: (ビットストリーム全体, payload のバイト数)．
    """
    encoder = constriction.stream.queue.RangeEncoder()
    escapes = []
    for k in range(hist.num_channels):
        idx, esc = hist._symbols(k, z[k].reshape(-1))
        encoder.encode(idx, constriction.stream.model.Categorical(hist.probs[k], perfect=False))
        escapes.append(esc)
    escapes = np.concatenate(escapes)
    if len(escapes) > 0:
        encoder.encode((escapes + ESCAPE_OFFSET).astype(np.int32), constriction.stream.model.Uniform(1 << ESCAPE_BITS))
    payload = encoder.get_compressed().tobytes()
    header = struct.pack(HEADER_FORMAT, height, width, model_id, setting_id)
    return header + payload, len(payload)


def decode(stream, hist, num_channels=None):
    """ビットストリームから量子化係数を復号する．

    Args:
        stream (bytes): encode() が出力したビットストリーム．
        hist (HistogramModel): 符号化時と同じ確率モデル．
        num_channels (int | None): 係数のチャネル数（省略時は hist から取得）．
    Returns:
        tuple[np.ndarray, dict]: (形状 (K, H/8, W/8) の整数配列, ヘッダ情報の辞書)．
    """
    height, width, model_id, setting_id = struct.unpack(HEADER_FORMAT, stream[:HEADER_BYTES])
    k_total = num_channels or hist.num_channels
    h, w = height // 8, width // 8
    words = np.frombuffer(stream[HEADER_BYTES:], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    idx = np.stack([decoder.decode(constriction.stream.model.Categorical(hist.probs[k], perfect=False), h * w)
                    for k in range(k_total)])
    escape_idx = (hist.hi - hist.lo + 1)[:k_total, None]
    is_escape = idx == escape_idx
    z = idx.astype(np.int64) + hist.lo[:k_total, None]
    num_escape = int(is_escape.sum())
    if num_escape > 0:
        values = decoder.decode(constriction.stream.model.Uniform(1 << ESCAPE_BITS), num_escape).astype(np.int64)
        z[is_escape] = values - ESCAPE_OFFSET
    info = {"height": height, "width": width, "model_id": model_id, "setting_id": setting_id}
    return z.reshape(k_total, h, w), info
