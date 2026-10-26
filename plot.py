"""
plot.py -- plots the modulated co-simulation results (with and without DPD).
Reads only, never simulates. The static AM/AM sweep is a separate operation
and lives in amam.py.

Two domains, labelled on every figure:
  true voltages ("tv") -- what ADS produces at the port, real dBm.
  normalized            -- each signal divided by its OWN RMS magnitude, per
                          dpd_modeling_MP.normalize_and_sync's definition of
                          "normalized" (v / rms(|v|)), not a display trick.

    ./adsenv.sh plot.py              plot and open
    ./adsenv.sh plot.py --no-show    plot only
"""
import os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ads_io as A
import dpd_modeling_MP as dpd


# ===========================================================================
# CONFIG
# ===========================================================================
# Both datasets come from the same schematic (DPA_Modulated_TB); they differ
# only in which stimulus file was fed to it, so both name their nets Vin/Vout.
REF_DS = "REF_run"       # driven by the plain LTE waveform
LIN_DS = "LIN_run"       # driven by the predistorted waveform

# ACPR bands in MHz. Each adjacent band is the 5 MHz channel slot directly
# abutting the carrier -- no guard gap -- so it captures the regrowth shoulder
# where it is strongest. BOTH sides are measured: a PA with memory produces
# asymmetric regrowth, so the lower and upper shoulders are not equal and
# reporting one alone understates the worse of the two.
ACPR_MAIN = (-2.5, 2.5)
ACPR_ADJ_LO = (-7.5, -2.5)
ACPR_ADJ_HI = (2.5, 7.5)

# Window for PSD_welch. hann matches DPD_modeling_MP.PSD_welch's default, and is the same window used in ADS's "Spectrum" block.
FFT_WINDOW = "hann"

# How much of the drive range the AM/AM plots show, measured down from the
# peak. The LTE envelope's nulls reach 55 dB below peak; plotting all of it
# squeezes the compression into a corner and it reads as a straight line.
AMAM_RANGE_DB = 25

# Width of the low-drive band the "ideal linear" reference is fitted over,
# measured up from the bottom of the plotted range. Must stay inside the
# region where the amplifier is still linear, or the reference picks up the
# very compression it exists to reveal.
SMALL_SIGNAL_SPAN_DB = 8

OPEN_WHEN_DONE = True
PLOT_DIR = Path(__file__).parent / "plots"
SAVED: list = []


# ===========================================================================
# REUSABLE PIECES
# ===========================================================================
def small_signal_gain(xp, yp, lo, span=SMALL_SIGNAL_SPAN_DB):
    """
    Gain over the lowest `span` dB of the plotted drive range, in dB.

    An ideal linear stage has slope exactly 1 in dB-dB -- double the input
    power, double the output power -- so the slope is not fitted, only the
    offset is. That offset IS the gain. Measuring it at the bottom of the
    drive range keeps the fit inside the region where the amplifier is still
    behaving linearly, so the reference line is not contaminated by the
    compression it is drawn to reveal.

    Works in either domain, since it is handed already-converted dB arrays.
    """
    band = (xp > lo) & (xp < lo + span)
    gain_db = float(np.mean(yp[band] - xp[band]))
    return gain_db


def normalized_db(v):
    """
    Real RMS normalization: v divided by its own RMS magnitude, then expressed
    as power in dB. This is dpd_modeling_MP.normalize_and_sync's definition of
    "normalized" -- v / rms(|v|) -- the same operation used to prepare data for
    the DPD model, not a display trick invented for this plot.

    A signal normalized this way has average power 0 dB by construction, so
    0 dB marks "at the signal's own typical level" -- it is not an absolute
    reference like dBm, and is labelled "dB rel. RMS" accordingly.
    """
    v = np.asarray(v)
    magnitude_rms = dpd.rms(np.abs(v))       # matches normalize_and_sync's own rms(np.abs(x))
    normalized_signal = v / magnitude_rms
    power_db = 10 * np.log10(np.abs(normalized_signal) ** 2 + 1e-30)
    return power_db


def psd(x):
    """Welch PSD. Returns (linear pxx, freq Hz)."""
    return dpd.PSD_welch(np.asarray(x), A.SR, FFT_WINDOW)


def band_power(pxx, freq, band):
    """Integrate a linear PSD over a band given in MHz."""
    f = np.asarray(freq) / 1e6
    m = (f >= band[0]) & (f <= band[1])
    bin_width_hz = abs(freq[1] - freq[0])
    power = float(np.sum(pxx[m]) * bin_width_hz)
    return power


def acpr(pxx, freq):
    """Lower and upper adjacent / main channel power, in dB."""
    main = band_power(pxx, freq, ACPR_MAIN)
    lower = band_power(pxx, freq, ACPR_ADJ_LO)
    upper = band_power(pxx, freq, ACPR_ADJ_HI)
    return (10 * np.log10(lower / main), 10 * np.log10(upper / main))


def acpr_worst(pxx, freq):
    """The worse (less negative) of the two adjacent channels."""
    return max(acpr(pxx, freq))


def align_inband(pxx, freq, pxx_ref):
    """
    Scale a PSD so its main-channel power matches the reference's, laying the
    flat tops on top of each other so the shoulders compare directly. ACPR is
    a ratio, so this changes what you see, not what you measure.
    """
    own = band_power(pxx, freq, ACPR_MAIN)
    ref_power = band_power(pxx_ref, freq, ACPR_MAIN)
    scale = ref_power / own
    aligned = pxx * scale
    return aligned


# ===========================================================================
# FIGURES
# ===========================================================================
def save(fig, name):
    """
    Save as PDF. matplotlib's PDF backend emits true vector output -- the
    curves stay as path operators, not an embedded bitmap -- so the figures
    scale cleanly in a LaTeX report.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = (PLOT_DIR / name).with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(path)              # no dpi: meaningless for vector output
    plt.close(fig)
    SAVED.append(path)
    print(f"    -> {path}")


def plot_amam(curves, domain, title, name):
    """
    Instantaneous AM/AM.

    domain "tv"         -- real dBm, exactly as measured (A.dbm).
    domain "normalized" -- each signal divided by its OWN RMS magnitude
                           (normalized_db), matching normalize_and_sync's
                           definition of "normalized".

    Each curve gets its own dashed "ideal linear" reference: slope 1, offset
    fitted from that curve's small-signal region (see small_signal_gain).
    Where the data peels away from its own dashed line is the compression.

    The x-axis covers AMAM_RANGE_DB down from the peak, because the envelope's
    nulls otherwise stretch it over 55 dB and flatten the curvature away.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    to_db = A.dbm if domain == "tv" else normalized_db

    hi = to_db(curves[0][0]).max() + 1
    lo = hi - AMAM_RANGE_DB - 1
    visible = []

    for i, (x, y, label) in enumerate(curves):
        xp, yp = to_db(x), to_db(y)
        ax.plot(xp, yp, ".", markersize=2, alpha=0.5, color=colors[i], label=label)
        visible.append(yp[(xp >= lo) & (xp <= hi)])

        # slope 1, offset = this curve's own small-signal gain
        gain = small_signal_gain(xp, yp, lo)
        ax.plot([lo, hi], [lo + gain, hi + gain], "--", linewidth=1.2,
                color=colors[i], alpha=0.85,
                label="ideal linear (small-signal fit)" if i == 0 else None)
        print(f"        small-signal gain, {label:34s} = {gain:+7.2f} dB")

    visible = np.concatenate(visible)
    ax.axvline(hi - 1, color="crimson", linestyle="--", linewidth=1.2)
    ax.axhline(visible.max(), color="seagreen", linestyle="--", linewidth=1.2)
    unit = "dBm" if domain == "tv" else "dB rel. RMS"
    print(f"      peak Pin = {hi - 1:.2f}   peak Pout = {visible.max():.2f}   ({unit})")

    pad = 0.06 * np.ptp(visible)
    ax.set_xlim(lo, hi)
    ax.set_ylim(visible.min() - pad, visible.max() + pad)
    ax.set_xlabel(f"Input Power ({unit})")
    ax.set_ylabel(f"Output Power ({unit})" if domain == "tv"
                  else f"Normalized Output Power ({unit})")
    ax.set_title(f"{title}\n[{'true voltages' if domain == 'tv' else 'normalized (RMS)'}]")
    ax.grid(True, alpha=0.4)
    for h in ax.legend(markerscale=6, loc="upper left", fontsize=9).legend_handles:
        h.set_alpha(1.0)
    save(fig, name)


def plot_spectrum(traces, title, name, acpr_of, align_to=None):
    """
    Spectra on one axes. `align_to` names the trace whose in-band level the
    others are laid onto. ACPR is always measured before any such scaling.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    spectra = [(label, style, *psd(sig)) for sig, label, style in traces]
    measured = {label: acpr(pxx, freq) for label, _, pxx, freq in spectra}

    ref = next((p for lb, _, p, _ in spectra if lb == align_to), None)
    for label, style, pxx, freq in spectra:
        shown = align_inband(pxx, freq, ref) if ref is not None else pxx
        ax.plot(freq / 1e6, 10 * np.log10(shown + 1e-30), style,
                linewidth=1.1, label=label)

    ax.axvspan(*ACPR_MAIN, color="tab:green", alpha=0.10)
    ax.axvspan(*ACPR_ADJ_LO, color="tab:red", alpha=0.10)
    ax.axvspan(*ACPR_ADJ_HI, color="tab:red", alpha=0.10)
    lo, hi = measured[acpr_of]
    ax.annotate(f"ACPR ({acpr_of})\nlower {lo:.2f} dB   upper {hi:.2f} dB",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right",
                va="bottom", bbox=dict(boxstyle="round", fc="lightyellow"))

    ax.set_xlim(-15, 15)
    ax.set_xlabel("Frequency offset from carrier (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(f"{title}\n[true voltages]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, name)
    return measured[acpr_of]


# ===========================================================================
# SECTIONS
# ===========================================================================
def section_1(d):
    print("\n=== SECTION 1 - without DPD ===")
    print("  1.1 Pout vs Pin, normalized")
    plot_amam([(d["vin"], d["vout"], "PA output (compressive)")],
              "normalized", "1.1  Pout vs Pin, no DPD",
              "1.1_pout_vs_pin_normalized.png")

    print("  1.2 Pout vs Pin, true voltages")
    plot_amam([(d["vin"], d["vout"], "PA output (compressive)")],
              "tv", "1.2  Pout vs Pin, no DPD", "1.2_pout_vs_pin_tv.png")

    print("  1.3 Spectrum, not aligned")
    a3 = plot_spectrum([(d["vin"], "PA input (LTE)", "-"),
                        (d["vout"], "PA output, no DPD", "-")],
                       "1.3  Spectrum, no DPD, not aligned",
                       "1.3_spectrum_tv_unaligned.png", "PA output, no DPD")

    print("  1.4 Spectrum, in-band aligned")
    a4 = plot_spectrum([(d["vin"], "PA input (LTE)", "-"),
                        (d["vout"], "PA output, no DPD", "-")],
                       "1.4  Spectrum, no DPD, in-band aligned (gain removed)",
                       "1.4_spectrum_tv_aligned.png", "PA output, no DPD",
                       align_to="PA input (LTE)")
    return {"1.3 no DPD, unaligned": a3, "1.4 no DPD, aligned": a4}


def section_2(d):
    print("\n=== SECTION 2 - with DPD ===")
    # x-axis is the original LTE drive, so the linearized output and the
    # predistorter's characteristic are both functions of the same input.
    # Every dataset is index aligned (measured PA lag = 0 samples).
    pin = d["vin"]
    curves = [(pin, d["vout_dpd"], "PA output, with DPD (linearized)"),
              (pin, d["vin_dpd"], "predistorted drive (expansive)"),
              (pin, d["vout"], "PA alone, no DPD (compressive)")]

    print("  2.1 Pout and predistorted vs Pin, true voltages")
    plot_amam(curves, "tv", "2.1  Pout and predistorted signal vs Pin, with DPD",
              "2.1_pout_and_predistorted_vs_pin_tv.png")

    print("  2.2 Pout and predistorted vs Pin, normalized")
    plot_amam(curves, "normalized",
              "2.2  Pout and predistorted signal vs Pin, with DPD",
              "2.2_pout_and_predistorted_vs_pin_normalized.png")

    outputs = [(d["vout_dpd"], "PA output, with DPD", "-"),
               (d["vout"], "PA output, no DPD (reference)", "--")]

    # 2.3 keeps the predistorted drive, sitting ~15 dB below the outputs, so
    # the expansion the DPD applied is visible next to the result it produced.
    print("  2.3 Spectrum, not aligned")
    a3 = plot_spectrum([(d["vin_dpd"], "predistorted input", "-")] + outputs,
                       "2.3  Spectrum, with DPD, not aligned",
                       "2.3_spectrum_tv_unaligned.png", "PA output, with DPD")

    # 2.4 drops it and compares the two PA outputs alone. Both runs were driven
    # at Pin = PIN_DBM and land within 0.02 dB of the same output power, so the
    # alignment barely moves anything -- it is the no-DPD trace that shifts,
    # since the target is the with-DPD output. No gain is removed here: that
    # only happens in 1.4, where an input is scaled up onto an output.
    print("  2.4 Spectrum, in-band aligned")
    a4 = plot_spectrum(outputs, "2.4  Spectrum, with DPD, in-band aligned",
                       "2.4_spectrum_tv_aligned.png", "PA output, with DPD",
                       align_to="PA output, with DPD")
    return {"2.3 with DPD, unaligned": a3, "2.4 with DPD, aligned": a4}


# ===========================================================================
def load():
    print("reading results ...")
    _, vin, vout = A.extract(REF_DS, axis="time")
    _, vin_dpd, vout_dpd = A.extract(LIN_DS, axis="time")
    print(f"  no DPD   : mean Pin={A.mean_dbm(vin):7.2f} dBm  "
          f"mean Pout={A.mean_dbm(vout):7.2f} dBm")
    print(f"  with DPD : mean Pin={A.mean_dbm(vin_dpd):7.2f} dBm  "
          f"mean Pout={A.mean_dbm(vout_dpd):7.2f} dBm")
    return dict(vin=vin, vout=vout, vin_dpd=vin_dpd, vout_dpd=vout_dpd)


def open_plots(paths):
    """
    Show the figures.

    The child gets a CLEANED environment: adsenv.sh sets LD_LIBRARY_PATH to
    ADS's lib dir and QT_QPA_PLATFORM=offscreen for this process, and a viewer
    inheriting either would render nothing or link against ADS's bundled Qt.

    A PDF viewer taking all the files at once is preferred; xdg-open is the
    last resort and gets the folder, since it would spawn a window per figure.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("LD_LIBRARY_PATH", "QT_QPA_PLATFORM", "MPLBACKEND")}
    viewer = next(v for v in ("papers", "evince", "okular", "xdg-open")
                  if shutil.which(v))
    args = [viewer, str(PLOT_DIR)] if viewer == "xdg-open" else [viewer, *map(str, paths)]
    subprocess.Popen(args, env=env, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nopened {len(paths)} figures in {viewer}")


def main():
    d = load()
    results = {**section_1(d), **section_2(d)}

    print("\n=== ACPR summary ===")
    print(f"  main {ACPR_MAIN[0]:+.1f}..{ACPR_MAIN[1]:+.1f} MHz, "
          f"lower {ACPR_ADJ_LO[0]:+.1f}..{ACPR_ADJ_LO[1]:+.1f} MHz, "
          f"upper {ACPR_ADJ_HI[0]:+.1f}..{ACPR_ADJ_HI[1]:+.1f} MHz\n")
    print(f"  {'':28s} {'lower':>9s} {'upper':>9s} {'worst':>9s}")
    for label, (lo, hi) in results.items():
        print(f"  {label:28s} {lo:8.2f} {hi:8.2f} {max(lo, hi):8.2f}  dB")
    ref_lo, ref_hi = results['1.3 no DPD, unaligned']
    lin_lo, lin_hi = results['2.3 with DPD, unaligned']
    print(f"\n  ACPR improvement : lower {ref_lo - lin_lo:+.2f} dB, "
          f"upper {ref_hi - lin_hi:+.2f} dB, "
          f"worst {max(ref_lo, ref_hi) - max(lin_lo, lin_hi):+.2f} dB")

    if OPEN_WHEN_DONE and "--no-show" not in sys.argv:
        open_plots(SAVED)


if __name__ == "__main__":
    main()
