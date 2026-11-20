"""CAE の学習（論文 3.2 節，Appendix A.4）．

order_001 の実験条件:
    - Optimizer: Adam，batch size 32，128x128 RGB パッチ
    - 学習率 1e-4，全係数を有効化した後 1e-5
    - 最大 1,000,000 更新（5,000 更新を 1 エポックとして 200 エポック）
    - Incremental Training（2 チャネルから開始し，改善が閾値未満になるたびに 1 チャネル追加）
    - 3 モデル: (α, チャネル数) = (0.01, 96), (0.05, 96), (0.2, 64)
    - 損失は Appendix A.4 の式 (16):
        L = α/N · (−ln q([f(x)∘λ]∘m + u)) + (1−α)/(1000·M) · ||x − g([f(x)∘λ]∘m/λ)||²
      N は係数の数，M は画素数の 3 倍．画素値は 0〜255．
推定仕様（論文に記載なし）:
    - Incremental Training の判定: 1,000 更新ごとに直近 1,000 更新の平均損失と，その前の 1,000 更新の
      平均損失を比較し，相対改善率が 1% 未満なら次のチャネルを有効化する．有効化直後の 2,000 更新は判定しない．
    - 1 エポック = 5,000 更新．エポックごとに検証・チェックポイント保存を行う．

実行:
    .venv_pytorch/bin/python programs/ex001_kodak_cae/train.py            # 3 モデルを並列学習（中断後は自動再開）
    .venv_pytorch/bin/python programs/ex001_kodak_cae/train.py --only 0.2_64
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import ROOT, load_dataloader, load_stats  # noqa: E402
from machine_learning_utils import ResultLogger  # noqa: E402
from model import CAE, load_model  # noqa: E402

EXPERIMENT = "ex001_kodak_cae"
OUTPUT_DIR = os.path.join(ROOT, "outputs", EXPERIMENT)

# ---------------------------------------------------------------------------
# 探索空間（order_001: 論文 Appendix A.4 の 3 モデル）と seed
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [(0.01, 96), (0.05, 96), (0.2, 64)]
SEEDS = [0]
BATCH_SIZE = 32
ITERS_PER_EPOCH = 5000
EPOCHS = 200  # 200 x 5,000 = 1,000,000 更新
LR_INITIAL = 1e-4
LR_FINAL = 1e-5
INITIAL_ENABLED = 2
INC_WINDOW = 1000
INC_THRESHOLD = 0.01
INC_WAIT = 2000
MAX_NONFINITE = 20
ITER_LOG_INTERVAL = 100


# ---------------------------------------------------------------------------
# 損失と評価指標
# ---------------------------------------------------------------------------
def loss_func(outputs, teacher_signals, alpha):
    """式 (16) のレート・歪み損失を計算する．

    Args:
        outputs (tuple): CAE.forward の戻り値 (x_hat (B,3,H,W), log_q (B,K,h,w), z (B,K,h,w))．
        teacher_signals (torch.Tensor): 原画像，形状 (B, 3, H, W)，値域 0〜255．
        alpha (float): レートと歪みのトレードオフ係数 α．
    Returns:
        dict[str, torch.Tensor]: "loss"（1 データあたりの損失），"rate_term"，"dist_term"（いずれもスカラー）．
    """
    x_hat, log_q, _ = outputs
    n_coef = log_q[0].numel()
    m_dim = teacher_signals[0].numel()
    rate = -log_q.flatten(1).sum(1) / n_coef
    dist = ((teacher_signals - x_hat) ** 2).flatten(1).sum(1) / (1000.0 * m_dim)
    loss = alpha * rate + (1.0 - alpha) * dist
    return {"loss": loss.mean(), "rate_term": rate.mean(), "dist_term": dist.mean()}


def metrics_func(outputs, teacher_signals):
    """MSE，PSNR，推定 bpp（雑音付き連続近似）を計算する．

    Args:
        outputs (tuple): CAE.forward の戻り値 (x_hat, log_q, z)．
        teacher_signals (torch.Tensor): 原画像，形状 (B, 3, H, W)．
    Returns:
        dict[str, float]: "mse", "psnr", "bpp_noisy", "nonzero"（非ゼロ係数の割合）の 1 データあたりの平均．
    """
    x_hat, log_q, z = outputs
    pixels = teacher_signals.shape[-1] * teacher_signals.shape[-2]
    mse = ((teacher_signals - x_hat.detach()) ** 2).flatten(1).mean(1)
    psnr = 10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))
    bpp = -log_q.detach().flatten(1).sum(1) / math.log(2.0) / pixels
    return {"mse": mse.mean().item(), "psnr": psnr.mean().item(), "bpp_noisy": bpp.mean().item(),
            "nonzero": (z.detach() != 0).float().mean().item()}


def iteration(model, inputs, teacher_signals, optimizer=None, alpha=0.01):
    """1 ミニバッチの学習または検証を行う．

    Args:
        model (torch.nn.Module): CAE．
        inputs (torch.Tensor): 入力画像，形状 (B, 3, 128, 128)．
        teacher_signals (torch.Tensor): 教師信号（入力と同一），形状 (B, 3, 128, 128)．
        optimizer (torch.optim.Optimizer | None): None でなければパラメータを更新する．
        alpha (float): 式 (16) の α．
    Returns:
        dict[str, float]: 損失の各項と評価指標の辞書．損失が非有限の場合は "finite"=0．
    """
    with torch.set_grad_enabled(optimizer is not None):
        outputs = model(inputs)
        losses = loss_func(outputs, teacher_signals, alpha)
    finite = bool(torch.isfinite(losses["loss"]).item())
    if optimizer is not None and finite:
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
    result = {k: v.item() for k, v in losses.items()}
    result.update(metrics_func(outputs, teacher_signals))
    result["finite"] = float(finite)
    return result


# ---------------------------------------------------------------------------
# Incremental Training
# ---------------------------------------------------------------------------
class IncrementalController:
    """式 (13) のマスク m を管理し，改善が閾値未満になったら係数（チャネル）を 1 つ有効化する．"""

    def __init__(self, num_channels, initial=INITIAL_ENABLED, window=INC_WINDOW, threshold=INC_THRESHOLD, wait=INC_WAIT):
        """状態を初期化する．

        Args:
            num_channels (int): 全チャネル数 K．
            initial (int): 初期に有効なチャネル数（論文では 2）．
            window (int): 平均損失を比較する更新数の窓幅．
            threshold (float): 次のチャネルを有効化する相対改善率の閾値．
            wait (int): 有効化直後に判定を行わない更新数．
        """
        self.num_channels = num_channels
        self.enabled = initial
        self.window, self.threshold, self.wait = window, threshold, wait
        self.buffer = []
        self.prev_mean = None
        self.since_enable = 0
        self.lr_reduced_at = None
        self.history = []  # (iteration, enabled)

    def mask(self, device):
        """現在のマスクを返す．

        Args:
            device (torch.device): 配置先デバイス．
        Returns:
            torch.Tensor: 形状 (K,) の 0/1 マスク．
        """
        m = torch.zeros(self.num_channels, device=device)
        m[:self.enabled] = 1.0
        return m

    @property
    def complete(self):
        """全チャネルが有効化済みかを返す．

        Returns:
            bool: 全チャネル有効なら True．
        """
        return self.enabled >= self.num_channels

    def step(self, loss, global_step):
        """1 更新分の損失を記録し，必要ならチャネルを有効化する．

        Args:
            loss (float): 当該更新の学習損失．
            global_step (int): 通算の更新数．
        Returns:
            bool: この更新でチャネルを追加した場合 True．
        """
        if self.complete:
            return False
        self.buffer.append(loss)
        self.since_enable += 1
        if len(self.buffer) < self.window:
            return False
        cur = float(np.mean(self.buffer))
        self.buffer = []
        prev, self.prev_mean = self.prev_mean, cur
        if prev is None or self.since_enable < self.wait:
            return False
        if (prev - cur) / abs(prev) >= self.threshold:
            return False
        self.enabled += 1
        self.since_enable = 0
        self.prev_mean = None
        self.history.append((global_step, self.enabled))
        return True

    def state_dict(self):
        """保存用の辞書を返す．

        Returns:
            dict: 状態の辞書．
        """
        return dict(self.__dict__)

    def load_state_dict(self, state):
        """状態を復元する．

        Args:
            state (dict): state_dict() の戻り値．
        Returns:
            None
        """
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# エポック・学習ループ
# ---------------------------------------------------------------------------
def epoch(model, dataloader, optimizer=None, alpha=0.01, controller=None, state=None, iter_log=None, position=0):
    """1 つの DataLoader の全データを学習または検証する．

    Args:
        model (torch.nn.Module): CAE．
        dataloader (torch.utils.data.DataLoader): データローダー．
        optimizer (torch.optim.Optimizer | None): None なら検証．
        alpha (float): 式 (16) の α．
        controller (IncrementalController | None): 学習時のマスク管理．
        state (dict | None): 学習時の通算状態（"global_step", "nonfinite"）．
        iter_log (list | None): 更新ごとのログ（ITER_LOG_INTERVAL ごとに追記）．
        position (int): tqdm の表示位置．
    Returns:
        dict[str, float]: 1 データあたりの損失・評価指標の平均．
    """
    device = next(model.parameters()).device
    model.train(optimizer is not None)
    totals, count = {}, 0
    bar = tqdm(dataloader, leave=False, position=position, desc="train" if optimizer else "val", dynamic_ncols=True)
    for inputs, teacher_signals in bar:
        inputs = inputs.to(device, non_blocking=True)
        teacher_signals = teacher_signals.to(device, non_blocking=True)
        result = iteration(model, inputs, teacher_signals, optimizer, alpha)
        if optimizer is not None:
            _after_train_step(model, optimizer, result, controller, state, iter_log)
        if result["finite"] == 0.0:
            continue
        n = inputs.shape[0]
        for k, v in result.items():
            totals[k] = totals.get(k, 0.0) + v * n
        count += n
        bar.set_postfix({k: f"{totals[k] / count:.4g}" for k in ["loss", "psnr", "bpp_noisy"]}
                        | ({"ch": controller.enabled, "lr": optimizer.param_groups[0]["lr"]} if controller else {}))
    return {k: v / max(count, 1) for k, v in totals.items()}


def _set_lr(optimizer, lr):
    """全パラメータグループの学習率を設定する．

    Args:
        optimizer (torch.optim.Optimizer): 最適化手法．
        lr (float): 学習率．
    Returns:
        None
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


def _after_train_step(model, optimizer, result, controller, state, iter_log):
    """1 更新後の処理（NaN 監視，Incremental Training，学習率の切替，更新ログ）．

    Args:
        model (torch.nn.Module): CAE．
        optimizer (torch.optim.Optimizer): 最適化手法．
        result (dict): iteration() の戻り値．
        controller (IncrementalController): マスク管理．
        state (dict): 通算状態．
        iter_log (list): 更新ログ．
    Returns:
        None
    """
    state["global_step"] += 1
    if result["finite"] == 0.0:
        state["nonfinite"] += 1
        state["nonfinite_total"] += 1
        if state["nonfinite"] > MAX_NONFINITE:
            raise FloatingPointError(f"損失が {MAX_NONFINITE} 回連続で非有限になった (step {state['global_step']})")
        return
    state["nonfinite"] = 0
    if controller.step(result["loss"], state["global_step"]):
        model.mask.copy_(controller.mask(model.mask.device))
    if controller.complete and controller.lr_reduced_at is None:
        controller.lr_reduced_at = state["global_step"]
        _set_lr(optimizer, LR_FINAL)
    if state["global_step"] % ITER_LOG_INTERVAL == 0:
        iter_log.append([state["global_step"], result["loss"], result["rate_term"], result["dist_term"],
                         result["psnr"], result["bpp_noisy"], optimizer.param_groups[0]["lr"], controller.enabled])


def _save_checkpoint(path, model, optimizer, controller, state, logger_history, best_val):
    """学習再開用のチェックポイントを一時ファイル経由で保存する．

    Args:
        path (str): 保存先．
        model (torch.nn.Module): CAE．
        optimizer (torch.optim.Optimizer): 最適化手法．
        controller (IncrementalController): マスク管理．
        state (dict): 通算状態．
        logger_history (dict): ResultLogger の履歴．
        best_val (float): これまでの最良検証損失．
    Returns:
        None
    """
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "controller": controller.state_dict(),
            "state": state, "logger": logger_history, "best_val": best_val,
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state()}}
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


def _log_line(target_dir, text):
    """ターミナルと学習ログファイルの両方に 1 行出力する．

    Args:
        target_dir (str): 結果保存先ディレクトリ．
        text (str): 出力する文字列．
    Returns:
        None
    """
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{os.path.basename(os.path.dirname(target_dir))}] {text}"
    tqdm.write(line)
    with open(os.path.join(target_dir, "train_stdout.log"), "a") as f:
        f.write(line + "\n")


def _metric_names():
    """ResultLogger に登録する指標名を返す．

    Returns:
        list[str]: 指標名のリスト．
    """
    keys = ["loss", "rate_term", "dist_term", "mse", "psnr", "bpp_noisy", "nonzero"]
    return ["epoch", "global_step", "lr", "enabled", "elapsed_sec"] + [f"train_{k}" for k in keys] + [f"val_{k}" for k in keys]


def train(target_dir, ModelClass, load_dataloader, epochs, batch_size, seed=0, hparams=None, position=0):
    """CAE を学習する．チェックポイントが存在すれば自動的に再開する．

    Args:
        target_dir (str): 結果保存先ディレクトリ．
        ModelClass (type): モデルのクラス (CAE)．
        load_dataloader (callable): DataLoader を返す関数．
        epochs (int): エポック数（1 エポック = ITERS_PER_EPOCH 更新）．
        batch_size (int): ミニバッチサイズ．
        seed (int): 乱数 seed．
        hparams (dict): {"alpha": float, "channels": int}．
        position (int): tqdm の表示位置．
    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    alpha, channels = hparams["alpha"], hparams["channels"]
    train_loader, val_loader = load_dataloader(seed, batch_size, samples_per_epoch=batch_size * ITERS_PER_EPOCH)
    stats = load_stats()
    model = load_model(ModelClass, seed=seed, model_kwargs={"latent_channels": channels, "mean": stats["mean"], "std": stats["std"]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [p for n, p in model.named_parameters() if n != "log_scales"]
    optimizer = torch.optim.Adam(trainable, lr=LR_INITIAL)
    controller = IncrementalController(channels)
    model.mask.copy_(controller.mask(device))
    logger = ResultLogger()
    logger.set_names(*_metric_names())
    state = {"global_step": 0, "epoch": 0, "nonfinite": 0, "nonfinite_total": 0, "elapsed_sec": 0.0}
    best_val, iter_log = float("inf"), []
    ckpt_path = os.path.join(target_dir, "checkpoint.pth")
    _write_config(target_dir, hparams, seed, epochs, batch_size)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        controller.load_state_dict(ckpt["controller"])
        state, best_val = ckpt["state"], ckpt["best_val"]
        logger.history = ckpt["logger"]
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
        np.random.set_state(ckpt["rng"]["numpy"])
        _log_line(target_dir, f"再開: epoch {state['epoch']}, step {state['global_step']}, enabled {controller.enabled}")
    else:
        # 0 エポック目（初期状態）の評価
        tr = epoch(model, train_loader, None, alpha, position=position)
        va = epoch(model, val_loader, None, alpha, position=position)
        _record(logger, 0, state, optimizer, controller, tr, va)
        _log_line(target_dir, f"epoch 0 (初期状態) val_loss {va['loss']:.5f} val_psnr {va['psnr']:.2f}")

    for ep in range(state["epoch"] + 1, epochs + 1):
        t0 = time.time()
        tr = epoch(model, train_loader, optimizer, alpha, controller, state, iter_log, position)
        va = epoch(model, val_loader, None, alpha, position=position)
        state["epoch"] = ep
        state["elapsed_sec"] += time.time() - t0
        _record(logger, ep, state, optimizer, controller, tr, va)
        _append_iter_log(target_dir, iter_log)
        if not math.isfinite(va["loss"]) or va["loss"] > 3.0 * best_val:
            _log_line(target_dir, f"警告: 検証損失が発散傾向 ({va['loss']:.5f}, best {best_val:.5f})")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        _save_checkpoint(ckpt_path, model, optimizer, controller, state, logger.history, best_val)
        logger.save(os.path.join(target_dir, "log.json"))
        _log_line(target_dir, f"epoch {ep}/{epochs} step {state['global_step']} lr {optimizer.param_groups[0]['lr']:.0e} "
                              f"ch {controller.enabled}/{channels} | train loss {tr['loss']:.5f} psnr {tr['psnr']:.2f} "
                              f"bpp {tr['bpp_noisy']:.3f} | val loss {va['loss']:.5f} psnr {va['psnr']:.2f} "
                              f"bpp {va['bpp_noisy']:.3f} | {time.time() - t0:.0f}s nonfinite {state['nonfinite_total']}")

    torch.save(model.state_dict(), os.path.join(target_dir, "final_model.pth"))
    with open(os.path.join(target_dir, "finished.json"), "w") as f:
        json.dump({"global_step": state["global_step"], "epochs": epochs, "lr_reduced_at": controller.lr_reduced_at,
                   "enable_history": controller.history, "elapsed_sec": state["elapsed_sec"]}, f, indent=2)
    _log_line(target_dir, "学習終了")


def _record(logger, ep, state, optimizer, controller, tr, va):
    """1 エポック分の学習・検証結果を ResultLogger に記録する．

    Args:
        logger (ResultLogger): ロガー．
        ep (int): エポック番号．
        state (dict): 通算状態．
        optimizer (torch.optim.Optimizer): 最適化手法．
        controller (IncrementalController): マスク管理．
        tr (dict): 学習データの指標．
        va (dict): 検証データの指標．
    Returns:
        None
    """
    keys = ["loss", "rate_term", "dist_term", "mse", "psnr", "bpp_noisy", "nonzero"]
    logger(ep, state["global_step"], optimizer.param_groups[0]["lr"], controller.enabled, state["elapsed_sec"],
           *[tr.get(k, float("nan")) for k in keys], *[va.get(k, float("nan")) for k in keys])


def _append_iter_log(target_dir, iter_log):
    """更新ごとのログを CSV に追記し，バッファを空にする．

    Args:
        target_dir (str): 結果保存先ディレクトリ．
        iter_log (list): 更新ログ（追記後に空にする）．
    Returns:
        None
    """
    path = os.path.join(target_dir, "iter_log.csv")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("step,loss,rate_term,dist_term,psnr,bpp_noisy,lr,enabled\n")
        for row in iter_log:
            f.write(",".join(str(v) for v in row) + "\n")
    iter_log.clear()


def _write_config(target_dir, hparams, seed, epochs, batch_size):
    """実行条件・乱数 seed・環境情報を config.json に保存する．

    Args:
        target_dir (str): 結果保存先ディレクトリ．
        hparams (dict): ハイパーパラメータ．
        seed (int): 乱数 seed．
        epochs (int): エポック数．
        batch_size (int): ミニバッチサイズ．
    Returns:
        None
    """
    config = {"hparams": hparams, "seed": seed, "epochs": epochs, "iters_per_epoch": ITERS_PER_EPOCH,
              "max_updates": epochs * ITERS_PER_EPOCH, "batch_size": batch_size, "effective_batch_size": batch_size,
              "gradient_accumulation": 1, "patch": 128, "optimizer": "Adam", "lr_initial": LR_INITIAL, "lr_final": LR_FINAL,
              "incremental": {"initial_enabled": INITIAL_ENABLED, "window": INC_WINDOW, "threshold": INC_THRESHOLD,
                              "wait": INC_WAIT},
              "dataset": "DIV2K train 800 (max side 1536), val: DIV2K valid 100 (200 fixed patches)",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "python": platform.python_version(), "tf32": True, "deterministic": True}
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def run_condition(args):
    """1 条件の学習を実行する（マルチプロセス用）．

    Args:
        args (tuple): (alpha, channels, seed, position, epochs)．
    Returns:
        str: 結果保存先ディレクトリ．
    """
    alpha, channels, seed, position, epochs = args
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    target_dir = os.path.join(OUTPUT_DIR, "CAE", f"{alpha}_{channels}", str(seed))
    if os.path.exists(os.path.join(target_dir, "finished.json")):
        print(f"skip (finished): {target_dir}")
        return target_dir
    train(target_dir, CAE, load_dataloader, epochs, BATCH_SIZE, seed, {"alpha": alpha, "channels": channels}, position)
    return target_dir


def run_parallel(func, jobs, processes):
    """非デーモンのプロセスで jobs を並列実行する（子プロセスが DataLoader のワーカを生成できるようにする）．

    Args:
        func (callable): 各ジョブで実行する関数（引数はジョブのタプル 1 つ）．
        jobs (list[tuple]): ジョブのリスト．
        processes (int): 同時実行数．
    Returns:
        list[int]: 各プロセスの終了コード．
    """
    ctx = mp.get_context("spawn")
    pending, running, codes = list(jobs), [], []
    while pending or running:
        while pending and len(running) < processes:
            p = ctx.Process(target=func, args=(pending.pop(0),))
            p.start()
            running.append(p)
        running[0].join()
        codes.append(running.pop(0).exitcode)
    return codes


def main():
    """全条件（MODEL_CONFIGS x SEEDS）を並列に学習する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="例: 0.2_64．指定した条件のみ学習する")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    jobs = []
    for pos, ((alpha, ch), seed) in enumerate(itertools.product(MODEL_CONFIGS, SEEDS)):
        if args.only and args.only != f"{alpha}_{ch}":
            continue
        jobs.append((alpha, ch, seed, pos, args.epochs))
    codes = run_parallel(run_condition, jobs, args.processes)
    print("終了コード:", codes)


if __name__ == "__main__":
    main()
