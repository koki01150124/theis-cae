"""データセットの作成と DataLoader の定義．

学習データ:
    論文の Flickr 434 枚は非公開のため，DIV2K の学習用 800 枚 (DIV2K_train_HR) を代替とする．
    論文に合わせ，各画像を 1536x1536 以下に縮小（アスペクト比を維持，Lanczos）し，可逆 PNG で保存する．
    学習時は 128x128 の RGB パッチをランダムに切り出す（論文に記載のないデータ拡張は行わない）．
検証データ:
    DIV2K の検証用 100 枚 (DIV2K_valid_HR) を同様に縮小し，seed 固定で 1 枚あたり 2 個の
    128x128 パッチを切り出した計 200 パッチを用いる．
評価データ:
    Kodak PhotoCD 24 枚 (kodim01〜24)．評価専用で，学習・ハイパーパラメータ探索には用いない．
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import set_seed  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW_DIR = os.path.join(ROOT, "datasets", "raw")
DATASET_DIR = os.path.join(ROOT, "datasets", "ex001_kodak_cae", "div2k_max1536")
KODAK_DIR = os.path.join(RAW_DIR, "kodak")
MAX_SIDE = 1536
PATCH = 128
VAL_PATCHES_PER_IMAGE = 2


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------
def _resize_to_max_side(img, max_side=MAX_SIDE):
    """長辺が max_side 以下になるようアスペクト比を保って縮小する．

    Args:
        img (PIL.Image.Image): RGB 画像．
        max_side (int): 許容する最大の辺の長さ．
    Returns:
        PIL.Image.Image: 縮小済み（または元のまま）の RGB 画像．
    """
    w, h = img.size
    ratio = max_side / max(w, h)
    if ratio >= 1.0:
        return img
    return img.resize((round(w * ratio), round(h * ratio)), Image.LANCZOS)


def _extract_and_resize(zip_path, out_dir):
    """DIV2K の zip を読み，縮小した PNG を out_dir に保存する．

    Args:
        zip_path (str): DIV2K_*_HR.zip のパス．
        out_dir (str): 保存先ディレクトリ．
    Returns:
        list[str]: 保存した PNG のパスのリスト（ソート済み）．
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".png"))
        for name in tqdm(names, desc=os.path.basename(zip_path), leave=False):
            out_path = os.path.join(out_dir, os.path.basename(name))
            if os.path.exists(out_path):
                continue
            with zf.open(name) as f:
                img = Image.open(f).convert("RGB")
            _resize_to_max_side(img).save(out_path)
    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def _pack_images(paths, npy_path, index_path):
    """画像群を 1 つの平坦な uint8 配列にまとめ，memmap で共有できる形式で保存する．

    Args:
        paths (list[str]): PNG のパスのリスト．
        npy_path (str): 連結した画素配列の保存先 (.npy)．
        index_path (str): 各画像の (offset, H, W) を保存する JSON のパス．
    Returns:
        None
    """
    shapes = [Image.open(p).size[::-1] for p in paths]
    total = sum(h * w * 3 for h, w in shapes)
    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(total,))
    index, offset = [], 0
    for p, (h, w) in tqdm(list(zip(paths, shapes)), desc="pack", leave=False):
        arr[offset:offset + h * w * 3] = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8).reshape(-1)
        index.append({"name": os.path.basename(p), "offset": offset, "h": h, "w": w})
        offset += h * w * 3
    arr.flush()
    with open(index_path, "w") as f:
        json.dump(index, f)


def _channel_stats(npy_path):
    """学習画像全体の RGB チャネルごとの平均と標準偏差を求める（固定の正規化に用いる）．

    Args:
        npy_path (str): 連結した画素配列 (.npy) のパス．
    Returns:
        dict: {"mean": [3 floats], "std": [3 floats]}（0〜255 の画素値空間）．
    """
    arr = np.load(npy_path, mmap_mode="r").reshape(-1, 3)
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for i in tqdm(range(0, arr.shape[0], 1 << 24), desc="stats", leave=False):
        chunk = arr[i:i + (1 << 24)].astype(np.float64)
        s += chunk.sum(0)
        s2 += (chunk ** 2).sum(0)
        n += chunk.shape[0]
    mean = s / n
    return {"mean": mean.tolist(), "std": np.sqrt(s2 / n - mean ** 2).tolist()}


def prepare_datasets():
    """DIV2K を縮小・連結し，正規化統計量とともに DATASET_DIR に保存する（作成済みならスキップ）．

    Returns:
        dict: 正規化統計量 {"mean": [...], "std": [...]}．
    """
    stats_path = os.path.join(DATASET_DIR, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    for split in ["train", "valid"]:
        paths = _extract_and_resize(os.path.join(RAW_DIR, "div2k", f"DIV2K_{split}_HR.zip"),
                                    os.path.join(DATASET_DIR, split))
        _pack_images(paths, os.path.join(DATASET_DIR, f"{split}.npy"), os.path.join(DATASET_DIR, f"{split}_index.json"))
    stats = _channel_stats(os.path.join(DATASET_DIR, "train.npy"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats():
    """保存済みの正規化統計量を読み込む．

    Returns:
        dict: {"mean": [3 floats], "std": [3 floats]}．
    """
    with open(os.path.join(DATASET_DIR, "stats.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PackedImages:
    """連結保存した画像群に memmap でアクセスするクラス（複数プロセスでページキャッシュを共有）．"""

    def __init__(self, split):
        """インデックスを読み込む（配列は遅延して memmap で開く）．

        Args:
            split (str): "train" または "valid"．
        """
        self.npy_path = os.path.join(DATASET_DIR, f"{split}.npy")
        with open(os.path.join(DATASET_DIR, f"{split}_index.json")) as f:
            self.index = json.load(f)
        self._arr = None

    def __len__(self):
        """画像数を返す．

        Returns:
            int: 画像数．
        """
        return len(self.index)

    def image(self, i):
        """i 番目の画像を返す．

        Args:
            i (int): 画像番号．
        Returns:
            np.ndarray: 形状 (H, W, 3) の uint8 配列（memmap のビュー）．
        """
        if self._arr is None:
            self._arr = np.load(self.npy_path, mmap_mode="r")
        e = self.index[i]
        return self._arr[e["offset"]:e["offset"] + e["h"] * e["w"] * 3].reshape(e["h"], e["w"], 3)


def _crop(img, top, left, size=PATCH):
    """画像から正方パッチを切り出し (3, size, size) の float テンソルにする．

    Args:
        img (np.ndarray): 形状 (H, W, 3) の uint8 配列．
        top (int): 切り出し開始の行．
        left (int): 切り出し開始の列．
        size (int): パッチの一辺．
    Returns:
        torch.Tensor: 形状 (3, size, size)，値域 0〜255 の float32 テンソル．
    """
    patch = np.ascontiguousarray(img[top:top + size, left:left + size])
    return torch.from_numpy(patch).permute(2, 0, 1).float()


class RandomPatchDataset(torch.utils.data.Dataset):
    """学習画像から 128x128 パッチを一様ランダムに切り出す Dataset．

    1 エポックを samples_per_epoch 個のパッチと定義する．画像はサンプラ側でシャッフルされ，
    切り出し位置はワーカごとに seed 固定した乱数で決める．
    """

    def __init__(self, split="train", samples_per_epoch=32 * 5000):
        """Dataset を構築する．

        Args:
            split (str): "train" または "valid"．
            samples_per_epoch (int): 1 エポックあたりのパッチ数．
        """
        self.images = PackedImages(split)
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(0)

    def __len__(self):
        """1 エポックのパッチ数を返す．

        Returns:
            int: パッチ数．
        """
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """idx に対応する画像からランダムなパッチを返す．

        Args:
            idx (int): サンプル番号．
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (入力, 教師信号)．いずれも形状 (3, 128, 128) の同一パッチ．
        """
        img = self.images.image(idx % len(self.images))
        top = int(self.rng.integers(0, img.shape[0] - PATCH + 1))
        left = int(self.rng.integers(0, img.shape[1] - PATCH + 1))
        x = _crop(img, top, left)
        return x, x


class FixedPatchDataset(torch.utils.data.Dataset):
    """検証画像から seed 固定で切り出したパッチの Dataset．"""

    def __init__(self, split="valid", per_image=VAL_PATCHES_PER_IMAGE, seed=0):
        """パッチ位置を決定する．

        Args:
            split (str): 対象のデータ分割．
            per_image (int): 1 枚あたりのパッチ数．
            seed (int): 切り出し位置を決める乱数 seed．
        """
        self.images = PackedImages(split)
        rng = np.random.default_rng(seed)
        self.items = []
        for i in range(len(self.images)):
            e = self.images.index[i]
            for _ in range(per_image):
                self.items.append((i, int(rng.integers(0, e["h"] - PATCH + 1)), int(rng.integers(0, e["w"] - PATCH + 1))))

    def __len__(self):
        """パッチ数を返す．

        Returns:
            int: パッチ数．
        """
        return len(self.items)

    def __getitem__(self, idx):
        """idx 番目のパッチを返す．

        Args:
            idx (int): パッチ番号．
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (入力, 教師信号)．形状 (3, 128, 128)．
        """
        i, top, left = self.items[idx]
        x = _crop(self.images.image(i), top, left)
        return x, x


def _worker_init(worker_id):
    """DataLoader のワーカごとに切り出し用乱数を seed 固定で初期化する．

    Args:
        worker_id (int): ワーカ番号．
    Returns:
        None
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def load_dataloader(seed=0, batch_size=32, samples_per_epoch=32 * 5000, num_workers=6):
    """学習用・検証用の DataLoader を作成する．

    Args:
        seed (int): シャッフルと切り出し位置の乱数 seed．
        batch_size (int): ミニバッチサイズ．
        samples_per_epoch (int): 学習用 DataLoader の 1 エポックあたりのパッチ数．
        num_workers (int): 学習用 DataLoader のワーカ数．
    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: (学習用, 検証用)．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator().manual_seed(seed)
    train_ds = RandomPatchDataset("train", samples_per_epoch)
    val_ds = FixedPatchDataset("valid", seed=seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=num_workers,
        worker_init_fn=_worker_init, drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def load_kodak():
    """Kodak PhotoCD 24 枚を読み込む（評価専用）．

    Returns:
        list[tuple[str, np.ndarray]]: (画像名, 形状 (H, W, 3) の uint8 配列) のリスト．
    """
    paths = sorted(glob.glob(os.path.join(KODAK_DIR, "kodim*.png")))
    return [(os.path.splitext(os.path.basename(p))[0], np.asarray(Image.open(p).convert("RGB"))) for p in paths]


if __name__ == "__main__":
    print(prepare_datasets())
