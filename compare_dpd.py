"""
compare_dpd.py -- memory polynomial vs LSTM predistorter, head to head.

A model that inverts the PA well on measured pairs has not been shown to
LINEARIZE anything. That is only settled by pushing its output back through
the circuit. This script does that for both predistorters and scores them
identically:

    1. reference ADS run, plain LTE stimulus            -> the "before"
    2. memory-polynomial DPD   (dpd_modeling_MP)        -> predistort, re-run
    3. LSTM DPD                (dpd_lstm_inverse)       -> predistort, re-run
    4. score all three the same way: NMSE and ACPR

Fairness, deliberately:
  * both models are fitted/trained on the SAME dataset;
  * both predistort the SAME stretch of LTE;
  * all three ADS runs use the same schematic, same Pin, same length;
  * all three are scored by the same functions against the same reference
    signal -- the original LTE, never the predistorted drive.

Nothing here modifies dpd_lstm_inverse.py, cosimulation.py,
dpd_modeling_MP.py, ads_io.py or plot.py; they are imported unchanged.

    ./adsenv.sh compare_dpd.py              run and open the figures
    ./adsenv.sh compare_dpd.py --no-show    run only
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import ads_io as A
import cosimulation as C          # MP path + nmse_db, unchanged
import dpd_lstm_inverse as L     # PALSTM + to_iq, from the predistorter file
import plot as P                  # acpr / psd, unchanged


# ===========================================================================
# CONFIG
# ===========================================================================
FIT_DATASET = "LSTM_train"     # 256k reference the LSTM was trained on
N_EVAL = 32_000                # samples actually pushed through ADS per run
PIN_DBM = C.PIN_DBM            # identical for every run, or the comparison
                               # is between two operating points
KA_LA = C.KA_LA                # memory polynomial order, same as the MP loop
CKPT = L.MODEL_DIR / "lstm_inverse_best.pt"
TAG = "compare"


# ===========================================================================
def predict_long(model, x_iq, device, chunk=8192):
    """
    Run the LSTM over a long sequence, carrying the hidden state across chunks.

    cuDNN refuses a single forward pass of a few hundred thousand timesteps
    (CUDNN_STATUS_NOT_SUPPORTED). Feeding it in chunks while threading (h, c)
    from one chunk into the next is mathematically identical to one continuous
    pass -- the recurrence is unbroken -- and bounds the memory. It lives here
    rather than in dpd_lstm_inverse.py because only deployment needs it:
    training never runs a sequence longer than one window.
    """
    model.eval()
    out, state = [], None
    with torch.no_grad():
        for i in range(0, len(x_iq), chunk):
            xb = torch.from_numpy(x_iq[i:i + chunk]).unsqueeze(0).to(device)
            h, state = model.lstm(xb, state)
            out.append(model.head(h).squeeze(0).cpu().numpy())
    a = np.concatenate(out)
    complex_signal = a[..., 0] + 1j * a[..., 1]
    return complex_signal


def lstm_predistort(stimulus, ckpt_path, device=None):
    """
    Deploy the trained post-inverse as a pre-inverse.

    The network was trained y_n -> x_n, where y_n is the gain-removed,
    peak-normalized PA output and x_n the peak-normalized PA input. Deployed,
    it is handed the WANTED signal in that same y_n domain and returns the
    drive: "if this were the output, what input produced it?"

    The stimulus is normalized to unit peak because that is exactly the range
    y_n covered in training -- feeding it hotter would ask the network to
    extrapolate, which is the LSTM equivalent of the amplitude ceiling that
    find_scaling_factor enforces for the memory polynomial.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob["config"]

    model = L.PALSTM(cfg["hidden"], cfg["layers"]).to(device)
    model.load_state_dict(blob["model"])

    wanted = stimulus / np.abs(stimulus).max()          # into the y_n domain
    drive = predict_long(model, L.to_iq(wanted), device)
    print(f"  LSTM predistorter: hidden={cfg['hidden']} layers={cfg['layers']}, "
          f"trained to {blob['val_nmse_db']:.2f} dB (epoch {blob['epoch']})")
    return drive, blob


def score(stimulus, pa_out, label):
    """NMSE against the wanted signal, plus ACPR of the output."""
    nmse = C.nmse_db(stimulus[:pa_out.size], pa_out)
    pxx, freq = P.psd(pa_out)
    lo, hi = P.acpr(pxx, freq)
    return {"label": label, "nmse_db": nmse,
            "acpr_lo_db": lo, "acpr_hi_db": hi, "acpr_db": max(lo, hi),
            "mean_pout_dbm": A.mean_dbm(pa_out)}


# ===========================================================================
def main():
    ws = A.open_workspace()

    # ---- 1. the data both models are fitted on ---------------------------
    print(f"loading fit data: {FIT_DATASET}")
    _, fit_in, fit_out = A.extract(FIT_DATASET, axis="time")
    fit_stim = A.read_tri(A.DATA / A.STIMULUS, n=fit_in.size)
    print(f"  {fit_in.size} samples")

    # ---- 2. memory polynomial -------------------------------------------
    print(f"\nfitting memory polynomial, ka/la = {KA_LA}")
    mp_pred_full, mp_lte_scaled = C.design_dpd(fit_in, fit_out, fit_stim, KA_LA)
    mp_pred = mp_pred_full[:N_EVAL]
    print(f"  predistorted {mp_pred.size} samples, "
          f"PAPR {20*np.log10(np.abs(mp_pred).max()/np.sqrt(np.mean(np.abs(mp_pred)**2))):.2f} dB")

    # ---- 3. LSTM ---------------------------------------------------------
    print("\nloading LSTM predistorter")
    lstm_pred, blob = lstm_predistort(fit_stim[:N_EVAL], CKPT)
    print(f"  predistorted {lstm_pred.size} samples, "
          f"PAPR {20*np.log10(np.abs(lstm_pred).max()/np.sqrt(np.mean(np.abs(lstm_pred)**2))):.2f} dB")

    # ---- 4. three ADS runs, identical settings ---------------------------
    runs = {}
    print(f"\nADS runs, {N_EVAL} samples each at Pin = {PIN_DBM} dBm")

    print("  [1/3] reference, no DPD")
    _, ref_in, ref_out = C.run_modulated(ws, A.STIMULUS, N_EVAL,
                                         f"{TAG}_ref", PIN_DBM)
    runs["none"] = (ref_in, ref_out)

    print("  [2/3] memory polynomial DPD")
    name = A.write_tri(mp_pred, f"{TAG}_mp.txt")
    _, mp_in, mp_out = C.run_modulated(ws, name, mp_pred.size, f"{TAG}_mp", PIN_DBM)
    runs["mp"] = (mp_in, mp_out)

    print("  [3/3] LSTM DPD")
    name = A.write_tri(lstm_pred, f"{TAG}_lstm.txt")
    _, ls_in, ls_out = C.run_modulated(ws, name, lstm_pred.size, f"{TAG}_lstm", PIN_DBM)
    runs["lstm"] = (ls_in, ls_out)

    A.close_workspace()

    # ---- 5. score, all against the same original LTE ---------------------
    stim = fit_stim[:N_EVAL]
    results = [score(stim, ref_out, "no DPD"),
               score(stim, mp_out, f"memory polynomial ({KA_LA[0]},{KA_LA[1]})"),
               score(stim, ls_out, f"LSTM ({blob['config']['hidden']} hidden)")]

    print("\n" + "=" * 68)
    print(f"{'':32s} {'NMSE':>10s} {'ACPR lo':>10s} {'ACPR hi':>10s} "
          f"{'mean Pout':>12s}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:32s} {r['nmse_db']:9.2f} dB {r['acpr_lo_db']:9.2f} dB "
              f"{r['acpr_hi_db']:9.2f} dB {r['mean_pout_dbm']:9.2f} dBm")
    print("-" * 80)
    base = results[0]
    for r in results[1:]:
        print(f"{r['label']:32s} {base['nmse_db']-r['nmse_db']:+9.2f} dB "
              f"{base['acpr_lo_db']-r['acpr_lo_db']:+9.2f} dB "
              f"{base['acpr_hi_db']-r['acpr_hi_db']:+9.2f} dB   (improvement)")
    print("=" * 80)

    # ---- 6. figures ------------------------------------------------------
    P.PLOT_DIR.mkdir(exist_ok=True)

    # Every trace is a PA OUTPUT; what differs is which predistorter drove it.
    # Spelling that out in the legend keeps "memory polynomial" from reading as
    # the model's own response rather than the amplifier's.
    # Model sizes are in the legend because the two counts are not in the same
    # units: the polynomial's coefficients are complex, the network's weights
    # are real, so "35 vs 4674" overstates the gap by a factor of two.
    n_mp = (KA_LA[0] * KA_LA[1])
    n_lstm = sum(v.numel() for v in blob["model"].values())
    LEGEND = {"none": "PA output, no DPD",
              "mp":   f"PA output, DPD = memory polynomial ({KA_LA[0]},{KA_LA[1]}), "
                      f"{n_mp} complex coef",
              "lstm": f"PA output, DPD = LSTM ({blob['config']['hidden']} hidden), "
                      f"{n_lstm} real params"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for (key, color, style) in [("none", "tab:green", "-"),
                                ("mp", "tab:blue", "-"),
                                ("lstm", "tab:orange", "--")]:
        label = LEGEND[key]
        pxx, freq = P.psd(runs[key][1])
        pxx = P.align_inband(pxx, freq, P.psd(runs["none"][1])[0])
        ax.plot(freq / 1e6, 10 * np.log10(pxx + 1e-30), style, color=color,
                linewidth=1.2, label=label)
    ax.axvspan(*P.ACPR_MAIN, color="tab:green", alpha=0.08)
    ax.axvspan(*P.ACPR_ADJ_LO, color="tab:red", alpha=0.08)
    ax.axvspan(*P.ACPR_ADJ_HI, color="tab:red", alpha=0.08)
    ax.set_xlim(-15, 15)
    ax.set_xlabel("Frequency offset from carrier (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("PA output spectrum -- no DPD vs memory polynomial vs LSTM\n"
                 "(in-band aligned; shaded = ACPR bands)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(P.PLOT_DIR / f"{TAG}_spectrum.pdf"); plt.close(fig)
    saved = [P.PLOT_DIR / f"{TAG}_spectrum.pdf"]

    # AM/AM through plot.plot_amam, so these read exactly like plot.py's 2.1
    # and 2.2: real dBm in the "tv" domain, and each signal divided by its own
    # RMS in the "normalized" one, with the same slope-1 small-signal
    # reference line and the same AMAM_RANGE_DB zoom.
    #
    # x is ref_in -- the MEASURED PA input of the no-DPD run, not the stimulus
    # file. ADS renormalises every modfile to Pin, so only the measured trace
    # carries true dBm; the file on disk is shape only.
    curves = [(ref_in, runs[k][1], LEGEND[k]) for k in ("none", "mp", "lstm")]
    for domain, suffix in (("tv", "tv"), ("normalized", "normalized")):
        name = f"{TAG}_amam_{suffix}.pdf"
        P.plot_amam(curves, domain,
                    "AM/AM of the linearized system\n(output vs the ORIGINAL signal)",
                    name)
        saved.append(P.PLOT_DIR / name)

    (L.MODEL_DIR / f"{TAG}_results.json").write_text(json.dumps(
        {"n_eval": N_EVAL, "pin_dbm": PIN_DBM, "fit_dataset": FIT_DATASET,
         "ka_la": list(KA_LA), "results": results}, indent=2))

    for f in saved:
        print(f"figure  {f}")
    print(f"results {L.MODEL_DIR / f'{TAG}_results.json'}")

    # plot.open_plots strips LD_LIBRARY_PATH / QT_QPA_PLATFORM from the child's
    # environment: adsenv.sh sets both for this process, and a viewer
    # inheriting either renders nothing or links against ADS's bundled Qt.
    if "--no-show" not in sys.argv:
        P.open_plots(saved)


if __name__ == "__main__":
    main()
