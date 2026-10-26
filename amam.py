"""
amam.py -- static AM/AM characterization of the amplifier.

A harmonic-balance sweep of DPA_AM_AM: one CW tone, stepped in power, showing
how the gain compresses as the amplifier is driven harder. This is a separate
operation from the modulated DPD loop -- no signal file, no DPD, no envelope --
so it lives in its own file.

    ./adsenv.sh amam.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ads_io as A


# ===========================================================================
# CONFIG
# ===========================================================================
POINTS = 101                 # HB1 Lin, the number of sweep points
DATASET = "AMAM_run"
PLOT_DIR = A.Path(__file__).parent / "plots"


def sweep(ws):
    """Run the Pin sweep and read back the AM/AM data."""
    design = A.open_design(ws, A.AMAM_CELL)
    design.instances["HB1"].parameters["Lin"].value = str(POINTS)
    A.simulate(design, DATASET)
    return A.extract(DATASET, axis="Pin")


def save(fig, ax, xlabel, ylabel, title, legend_loc, name):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n[true voltages]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc=legend_loc, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name)          # PDF: vector, for LaTeX
    plt.close(fig)
    print(f"    -> {PLOT_DIR / name}")


def plot(vin, vout):
    """Two figures: the AM/AM curve, and gain vs drive."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    p_in, p_out = A.dbm(vin), A.dbm(vout)
    gain = p_out - p_in

    # small-signal gain, from the bottom of the sweep where it is still linear
    g0 = float(np.mean(gain[:max(3, len(gain) // 10)]))

    # 1 dB compression point: the first swept point whose gain has fallen a
    # full dB below the small-signal value. Marked on BOTH figures -- it is the
    # operating point the modulated runs are set against, so it should be
    # readable from the curve and not only from the caption.
    below = np.where(gain <= g0 - 1)[0]
    p1_in = p1_out = None
    if below.size:
        i = below[0]
        p1_in, p1_out = float(p_in[i]), float(p_out[i])
        print(f"      1 dB compression at Pin = {p1_in:.2f} dBm, "
              f"Pout = {p1_out:.2f} dBm")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(p_in, p_out, "-", linewidth=2, label="PA output")
    ax.plot(p_in, p_in + g0, "--", linewidth=1.2, color="tab:orange",
            label=f"ideal linear ({g0:.2f} dB gain)")
    if p1_in is not None:
        ax.plot([p1_in], [p1_out], "o", color="crimson", markersize=8,
                zorder=5, label=f"1 dB compression ({p1_in:.2f}, {p1_out:.2f}) dBm")
        ax.annotate(f"$P_{{1dB}}$\n{p1_in:.2f} dBm in\n{p1_out:.2f} dBm out",
                    xy=(p1_in, p1_out), xytext=(-70, 18),
                    textcoords="offset points", fontsize=9, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1))
    save(fig, ax, "Input Power (dBm)", "Output Power (dBm)",
         "AM/AM -- harmonic-balance sweep", "upper left", "amam_pout_vs_pin.pdf")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(p_in, gain, "-", linewidth=2)
    ax.axhline(g0, color="tab:orange", linestyle="--", linewidth=1.2,
               label=f"small-signal gain {g0:.2f} dB")
    ax.axhline(g0 - 1, color="crimson", linestyle=":", linewidth=1.2,
               label=f"1 dB compression ({g0 - 1:.2f} dB)")
    if p1_in is not None:
        ax.axvline(p1_in, color="crimson", linestyle="--", linewidth=1.2)
        ax.plot([p1_in], [g0 - 1], "o", color="crimson", markersize=8,
                zorder=5, label=f"$P_{{1dB}}$ = {p1_in:.2f} dBm")
        ax.annotate(f"{p1_in:.2f} dBm", xy=(p1_in, g0 - 1), xytext=(-78, -26),
                    textcoords="offset points", fontsize=9, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1))
    save(fig, ax, "Input Power (dBm)", "Gain (dB)",
         "Gain vs drive", "lower left", "amam_gain_vs_pin.pdf")

    print(f"\n  swept {len(p_in)} points, Pin {p_in.min():.1f} .. {p_in.max():.1f} dBm")
    print(f"  gain {gain.min():.2f} .. {gain.max():.2f} dB  "
          f"({gain.max() - gain.min():.2f} dB of compression)")


if __name__ == "__main__":
    ws = A.open_workspace()
    print(f"AM/AM sweep of {A.AMAM_CELL}, {POINTS} points")
    _, vin, vout = sweep(ws)
    plot(vin, vout)
    A.close_workspace()
