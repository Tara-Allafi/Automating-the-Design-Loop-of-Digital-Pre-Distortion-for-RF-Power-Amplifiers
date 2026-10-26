"""
cosimulation.py -- the ADS <-> Python DPD co-simulation loop.

    reference run -> identify DPD -> inject -> linearized run -> score

Both ADS runs use the SAME schematic, DPA_Modulated_TB. Only the stimulus file
changes: the plain LTE waveform for the reference run, the predistorted
waveform for the linearized one. A second testbench is unnecessary -- checked
against the netlists, DPD_TB is the same amplifier, bias and topology.

    ./adsenv.sh cosimulation.py
"""
import numpy as np

import ads_io as A
import dpd_modeling_MP as dpd


# ===========================================================================
# CONFIG
# ===========================================================================
N_SAMPLES = 6400         # truncate both runs; the full LTE file is 1 228 800
PIN_DBM = 12.19          # drive level, identical for both runs or the
                         # comparison is between two operating points
LABEL = "run"            # dataset names: REF_<label>, LIN_<label>

# ka = 7 amplitude powers, la = 5 memory taps -> 35 coefficients. Chosen by
# manual closed-loop testing, not automatically; the shortlist it was picked
# from and the measurements that settled it are in the docstring of
# dpd_modeling_MP.shortlist_orders.
KA_LA = (7, 5)


# ===========================================================================
def run_modulated(ws, signal_file, n_samples, dataset_name, pin_dbm=PIN_DBM):
    """Drive the envelope testbench with a given stimulus file."""
    design = A.open_design(ws, A.MOD_CELL)
    A.set_vars(design, A.envelope_vars(n_samples, pin_dbm))
    A.set_modfile(design, signal_file)
    A.simulate(design, dataset_name)
    return A.extract(dataset_name, axis="time")


def design_dpd(pa_in, pa_out, stimulus, ka_la=KA_LA):
    """Fit the amplifier's inverse and predistort the stimulus."""
    pa_in_n, pa_out_n, lte_n, *_ = dpd.normalize_and_sync(
        pa_in, pa_out, stimulus, A.BW, A.SR)

    # (ka, la, paIn, paOut) fits the INVERSE -- see dpd_modeling_MP.
    model = dpd.get_MP_model(ka_la[0], ka_la[1], pa_in_n, pa_out_n)

    # Scale the drive so the predistorter's expanded peaks stay inside the
    # amplitude range the model was actually fitted over.
    _, lte_scaled, predistorted = dpd.find_scaling_factor(
        lte_n, model, float(np.max(np.abs(pa_in_n))))
    return np.asarray(predistorted), np.asarray(lte_scaled)


def nmse_db(stimulus, output):
    """
    How far the output is from a purely linear response to the stimulus.

    Compared against what we WANTED (the original signal), never against what
    was fed into the PA -- the amplifier was never linear in its own input.
    Bulk delay and complex gain are removed first, so what remains is only the
    nonlinear distortion.
    """
    n = min(stimulus.size, output.size)
    ref, mes = stimulus[:n], output[:n]

    lag = int(np.argmax(np.abs(np.correlate(mes, ref, "full"))) - (n - 1))
    if lag > 0:
        ref, mes = ref[:n - lag], mes[lag:]
    elif lag < 0:
        ref, mes = ref[-lag:], mes[:n + lag]

    gain = np.vdot(ref, mes) / np.vdot(ref, ref)
    error_power = np.sum(np.abs(mes - gain * ref) ** 2)
    signal_power = np.sum(np.abs(mes) ** 2)
    nmse = float(10 * np.log10(error_power / signal_power))
    return nmse


def loop(ws, n_samples=N_SAMPLES, label=LABEL, ka_la=KA_LA):
    """One full pass of the co-simulation loop."""
    print("1/4 reference run   (LTE stimulus)")
    _, pa_in, pa_out = run_modulated(ws, A.STIMULUS, n_samples, f"REF_{label}")

    print("2/4 stimulus")
    stimulus = A.read_tri(A.DATA / A.STIMULUS, n=n_samples)

    print("3/4 DPD identification")
    predistorted, lte_scaled = design_dpd(pa_in, pa_out, stimulus, ka_la)

    print("4/4 linearized run  (predistorted stimulus, same schematic)")
    name = A.write_tri(predistorted, f"PRED_{label}.txt")
    _, lin_in, lin_out = run_modulated(ws, name, predistorted.size, f"LIN_{label}")

    before, after = nmse_db(lte_scaled, pa_out), nmse_db(lte_scaled, lin_out)
    print(f"\ndistortion vs wanted : {before:.2f} dB -> {after:.2f} dB "
          f"({before - after:+.2f} dB)")
    print(f"mean output power    : {A.mean_dbm(pa_out):.2f} -> "
          f"{A.mean_dbm(lin_out):.2f} dBm")
    return pa_in, pa_out, lin_in, lin_out, lte_scaled


if __name__ == "__main__":
    ws = A.open_workspace()
    loop(ws)
    A.close_workspace()
