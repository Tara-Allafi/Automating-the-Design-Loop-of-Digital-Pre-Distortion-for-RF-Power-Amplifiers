"""
dpd_lstm_inverse.py -- LSTM PREDISTORTER, trained by indirect learning.

    PA            x -> y      the amplifier, as measured in ADS
    post-inverse  y -> x      what is trained here
    deployed as   x -> z      z drives the PA, so PA(z) ~ G*x

Fit the inverse straight from measured pairs by swapping input and target. No
forward PA model is needed anywhere. compare_dpd.py imports PALSTM and to_iq
from here to deploy the trained network.

    ./adsenv.sh dpd_lstm_inverse.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ads_io as A
import dpd_modeling_MP as dpd     # xcorr, reused rather than reimplemented


# ===========================================================================
# CONFIG
# ===========================================================================
DATASET = "LSTM_train"       # ADS dataset in the workspace data dir
SEQ_LEN, STRIDE = 128, 32    # window length and hop; stride < length overlaps
WARMUP = 16                  # leading samples of each window excluded from the
                             # loss, where the hidden state is still meaningless
HIDDEN, LAYERS = 32, 1       # small on purpose: a big net memorises the record
LR, BATCH_SIZE = 1e-3, 64
EPOCHS, PATIENCE = 200, 20   # patience = epochs without val improvement
CLIP, SEED = 1.0, 0
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = Path(__file__).parent
PLOT_DIR, MODEL_DIR = HERE / "plots", HERE / "models"


# ===========================================================================
# STEPS 1-4  --  raw ADS data to normalised I/Q
# ===========================================================================
def load_and_preprocess():
    _, x, y = A.extract(DATASET, axis="time")
    print(f"loading {DATASET}: {x.size} samples at {A.SR/1e6:.2f} MHz")

    # Alignment FIRST: the gain estimate and the x[n]<->y[n] pairing are both
    # wrong if the output is still sitting a few samples late.
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    lag = int(dpd.xcorr(np.abs(y), np.abs(x)))   # |.| so carrier phase cannot
    if lag > 0:                                  # smear the correlation peak
        x, y = x[:n - lag], y[lag:]
    elif lag < 0:
        x, y = x[-lag:], y[:n + lag]

    # Remove the constant complex gain, so the net learns the DISTORTION
    # rather than relearning ~15 dB of flat gain and a fixed phase rotation.
    gain = np.vdot(x, y) / np.vdot(x, x)
    y = y / gain

    # Unit PEAK, not unit RMS: the LSTM's sigmoid/tanh gates saturate outside
    # [-1, 1]. (normalize_and_sync divides by RMS, leaving peaks near 2.8.)
    sx, sy = float(np.abs(x).max()), float(np.abs(y).max())
    x, y = x / sx, y / sy

    print(f"  lag={lag}  gain={np.abs(gain):.3f}/_{np.degrees(np.angle(gain)):+.1f}deg"
          f"  peaks normalised (sx={sx:.3f} sy={sy:.3f})")
    return x, y, {"lag": lag, "gain_real": float(gain.real),
                  "gain_imag": float(gain.imag), "scale_x": sx, "scale_y": sy}


def to_iq(z):
    """Complex -> two real channels, (N, 2)."""
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def to_complex(t):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a[..., 0] + 1j * a[..., 1]


# ===========================================================================
# STEPS 6-8  --  signals to batched windows
#
# Step 5, the direction swap, happens at this function's CALL SITE in main().
# ===========================================================================
def make_loaders(net_in, net_target):
    # CONTIGUOUS time segments, never a random split: stride < seq_len means
    # neighbouring windows share samples, so a random split would leak train
    # data into validation and flatter the model.
    n = len(net_in)
    i, j = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    splits = [(net_in[:i], net_target[:i]), (net_in[i:j], net_target[i:j]),
              (net_in[j:], net_target[j:])]

    sets = {}
    for name, (a, b) in zip(("train", "val", "test"), splits):
        a, b = to_iq(a), to_iq(b)
        # x and y sliced with the SAME start index, so each example is one
        # contiguous chronological pair. Raw samples are never permuted.
        starts = range(0, len(a) - SEQ_LEN + 1, STRIDE)
        sets[name] = (torch.from_numpy(np.stack([a[s:s + SEQ_LEN] for s in starts])),
                      torch.from_numpy(np.stack([b[s:s + SEQ_LEN] for s in starts])))
        print(f"  {name:5s} {sets[name][0].shape[0]:5d} windows of {SEQ_LEN}")

    # shuffle=True is safe ONLY here: it permutes whole windows, each of which
    # still holds its own samples in chronological order.
    return (DataLoader(TensorDataset(*sets["train"]), batch_size=BATCH_SIZE, shuffle=True),
            {n: DataLoader(TensorDataset(*sets[n]), batch_size=BATCH_SIZE) for n in sets},
            splits)


# ===========================================================================
# STEP 9  --  the model
# ===========================================================================
class PALSTM(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


# ===========================================================================
# STEPS 10-11  --  train, keep the best checkpoint
# ===========================================================================
def windows_nmse(model, loader, device):
    """
    NMSE in dB over a loader, post-warm-up samples only.

    Predictions and targets are gathered first and scored in ONE dpd.nmse call,
    reusing the memory-polynomial path's formula rather than a second copy of
    it. Scoring per batch and averaging would average decibels, which is wrong;
    one call over the concatenation is exact. The arrays are ~1 MB.
    """
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred.append(to_complex(model(xb.to(device))[:, WARMUP:, :].cpu()).ravel())
            true.append(to_complex(yb[:, WARMUP:, :]).ravel())
    return float(dpd.nmse(np.concatenate(true), np.concatenate(pred)))


def train(train_loader, eval_loaders, tag, meta, device):
    """Adam + early stopping on validation NMSE; best checkpoint kept."""
    model = PALSTM().to(device)
    print(f"\nLSTM({HIDDEN}x{LAYERS}) + Linear, "
          f"{sum(p.numel() for p in model.parameters())} parameters on {device}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    hist = {"epoch": [], "train_nmse": [], "val_nmse": []}
    best, best_epoch, bad = np.inf, -1, 0
    ckpt = MODEL_DIR / f"{tag}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            # [:, WARMUP:, :] here and in every metric, so training and
            # evaluation agree on which samples count.
            lossf(model(xb)[:, WARMUP:, :], yb[:, WARMUP:, :]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        tr = windows_nmse(model, eval_loaders["train"], device)
        va = windows_nmse(model, eval_loaders["val"], device)
        for k, v in zip(hist, (epoch, tr, va)):
            hist[k].append(v)

        improved = va < best - 1e-4
        if improved:
            best, best_epoch, bad = va, epoch, 0
            torch.save({"model": model.state_dict(), "val_nmse_db": va,
                        "epoch": epoch, "config": {"hidden": HIDDEN, "layers": LAYERS,
                        "seq_len": SEQ_LEN, "stride": STRIDE, "warmup": WARMUP,
                        "lr": LR, "seed": SEED, "dataset": DATASET}, **meta}, ckpt)
        else:
            bad += 1
        if improved or epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}   train {tr:7.2f} dB   val {va:7.2f} dB"
                  f"{'  *' if improved else ''}")
        if bad >= PATIENCE:
            print(f"  early stop: no val improvement for {PATIENCE} epochs")
            break

    model.load_state_dict(torch.load(ckpt)["model"])
    print(f"\nbest epoch {best_epoch}")
    for name in ("train", "val", "test"):
        print(f"  {name:5s} NMSE = {windows_nmse(model, eval_loaders[name], device):7.2f} dB")

    np.savez(MODEL_DIR / f"{tag}_history.npz", **{k: np.asarray(v) for k, v in hist.items()})
    plot_history(hist, tag)
    print(f"checkpoint {ckpt}")
    return model, hist, best_epoch, best


# ===========================================================================
# STEP 12  --  report
# ===========================================================================
def plot_history(hist, tag):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist["epoch"], hist["train_nmse"], label="train")
    ax.plot(hist["epoch"], hist["val_nmse"], label="validation")
    b = int(np.argmin(hist["val_nmse"]))
    ax.axvline(hist["epoch"][b], color="crimson", linestyle="--", linewidth=1,
               label=f"best epoch ({hist['val_nmse'][b]:.2f} dB)")
    ax.set_xlabel("epoch"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"{tag} -- training history")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{tag}_history.pdf")
    plt.close(fig)


# ===========================================================================
# THE PIPELINE
#
# Step 5 is one line: make_loaders(y, x) instead of make_loaders(x, y). That
# swap is the only thing separating this predistorter from a forward PA model
# -- every other step is direction-agnostic.
# ===========================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    x, y, preproc = load_and_preprocess()
    print("  direction  y -> x  (post-inverse / predistorter)")
    train_loader, eval_loaders, _ = make_loaders(y, x)      # <- step 5
    train(train_loader, eval_loaders, "lstm_inverse",
          {"preproc": preproc, "direction": "post-inverse (y -> x)"},
          torch.device(DEVICE))

    print("\nNOTE: this NMSE says how well the network INVERTS the PA on measured")
    print("      pairs. Whether it LINEARIZES is settled only by pushing its")
    print("      output back through ADS -- that is what compare_dpd.py does.")


if __name__ == "__main__":
    main()
