"""
dpd_modeling_MP.py -- the DPD algorithms this project needs, self-contained.

Extracted from the XLIM / Rapid.Space DPD_modeling code (via AST, so the
numerics are byte-identical): nmse, algoLMS, xcorr, correlation,
coarsePhaseCorrection, estimGMPopt, identif_dpd_GMP, simGMPopt, rms,
_init_mp_model, get_MP_model, PSD_welch, normalize_and_sync,
find_scaling_factor. shortlist_orders at the bottom is new.

The upstream order search was NOT extracted: it scored candidates on the same
data they were fitted to, which always rewards more coefficients and lands on
inverses that are numerically explosive. shortlist_orders replaces it.

THE MODEL -- a memory polynomial:

    y[n] = sum over k,l of  coef[k,l] * |x[n-l]|^k * x[n-l]

ka = number of amplitude powers, la = memory depth. _build_pphi makes one
column per (k,l), algoLMS solves for the coefficients by regularized least
squares, simGMPopt applies a fitted model to a new signal.

ARGUMENT ORDER. get_MP_model(ka, la, x, y) fits x -> y. For a DPD pass
(ka, la, paIn, paOut): identif_dpd_GMP -> estimGMPopt swaps them internally, so
what is fitted is paOut -> paIn, the amplifier's INVERSE. Passing them the
other way fits the forward PA model and roughly doubles the distortion.
"""
import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.signal import get_window, welch


# ===========================================================================
# LAYER 1 -- CONDITIONING THE MEASURED DATA
#
# Before anything can be fitted, the measured input/output pair has to be made
# comparable. Gain, delay and phase rotation are what a *linear* system does to
# a signal; strip all three and whatever difference survives is the
# nonlinearity, which is the only thing the model should have to learn.
#
# This layer is also where the pipeline's second-largest cost sits: the four
# signal.resample calls in `correlation` are 0.58 s of it at 256k samples.
# ===========================================================================

def rms(array):
    
   return np.sqrt(np.mean(array ** 2))


# NOTE: rms squares the array directly, not |.|^2, so it is only correct on a
# real non-negative input. Every call site hands it np.abs(x) first.


def xcorr(x, y):
    """Lag (in samples) at the peak of the cross-correlation of x against y."""
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    lag = lags[np.argmax(corr)]

    return lag


def correlation(tx, rx):
    """
    Performs tx/rx synchronization and signal alignement
    tx    : transmitted signal
    rx    : received signal

    returns
    txSig, rxSig  : synchonized tx and rx signals
    """


    # The true PA in/out delay is generally a fraction of a sample, so both
    # signals go up 16x first: that buys the lag search a resolution of 1/16
    # of a sample. These four resamples are the expensive part of the file --
    # 0.58 s of the pipeline at 256k samples.
    # Resample transmit waveform
    osr = 16  # Another upSampling for best synchronization
    tx = signal.resample(tx, len(tx) * osr)
    rx = signal.resample(rx, len(rx) * osr)

    # Correlate the magnitudes, not the complex signals: carrier phase would
    # otherwise smear the correlation peak.
    lag = xcorr(np.abs(rx), np.abs(tx))

    # Data alignement and downSampling
    start = 0

    # Roll rx by the lag (wrapping the tail around), then back down to the
    # original rate.
    txSig = tx[start:len(tx)]
    rxSig = np.concatenate((rx[lag+start:start + len(txSig)], rx[0: lag]))
    txSig = signal.resample(txSig, len(txSig) // osr)
    rxSig = signal.resample(rxSig, len(rxSig) // osr)

    return txSig, rxSig


def coarsePhaseCorrection(x, y, bw, fs):

    # Removes the constant phase rotation that `correlation` leaves behind:
    # delay was taken out in samples, this takes out what remains in phase.

    # Coarse phase correction

    # commpute fft of x and y
    X = fft(x)
    Y = fft(y)
    # Compute BW index
    BW_bin = int(bw * len(X) / fs)
    # Compute phase difference
    angle = np.unwrap(np.angle(Y/X))


    # Averaged over in-band bins only -- callers pass bw/6, i.e. the middle
    # sixth of the band, where the SNR is best.
    # Compute avg angle over the BW
    avgAngle = np.mean(angle[:BW_bin])

    # Apply phase correction
    y = y * np.exp(-1j * avgAngle)

    return y


def normalize_and_sync(paIn, paOut, LTE, bw, sample_rate):
        """
        Normalize signals (RMS) and perform time/phase synchronization.
        Returns:
            paIn, paOut, LTE: normalized signals
            paIn_notNorm, paOut_notNorm, LTE_notNorm: unnormalized signals
        """
        # Layer 1's public entry point: gain out (RMS), delay out
        # (correlation), phase out (coarsePhaseCorrection), in that order.
        rms_pain = rms(np.abs(paIn))
        rms_paout = rms(np.abs(paOut))
        rms_lte = rms(np.abs(LTE))

        # Unit average power on each signal -- this is the gain removal.
        paIn_norm = paIn / rms_pain
        paOut_norm = paOut / rms_paout
        LTE_norm = LTE / rms_lte

        # Time and Phase synchronization IN/OUT of the PA
        paIn_sync, paOut_sync = correlation(paIn_norm, paOut_norm)
        paOut_sync = coarsePhaseCorrection(paIn_sync, paOut_sync, bw/6, sample_rate)

        # The _notNorm returns undo the RMS scaling on the *synchronized*
        # signals, so callers can plot in real units without redoing the sync.
        paIn_notNorm = paIn_sync * rms_pain
        paOut_notNorm = paOut_sync * rms_paout
        LTE_notNorm = LTE_norm * rms_lte

        return paIn_sync, paOut_sync, LTE_norm, paIn_notNorm, paOut_notNorm, LTE_notNorm


# ===========================================================================
# LAYER 2 -- THE MODEL
#
# The parameter dict and the regressor matrix. Once PPHI exists the whole
# nonlinear model is LINEAR in its coefficients -- y = PPHI @ c -- which is the
# trick that makes memory polynomials fittable by least squares, and the reason
# this matrix gets built twice per pipeline run (once to fit, once to apply).
# ===========================================================================

def _init_mp_model(ka: int, la: int) -> dict:
    return {"parameters": {"Ka": ka, "La": la}}


def _build_pphi(x, gmpModel):
    """
    The regressor matrix: one row per time sample, one column per (k, l) term,
    holding |x[n-l]|^k * x[n-l]. Shared by estimGMPopt (which fits against it)
    and simGMPopt (which applies fitted coefficients to it).

    Once this exists the model is LINEAR in the coefficients -- y = PPHI @ c --
    which is what makes a memory polynomial fittable by least squares.
    """
    Ka, La = gmpModel['parameters']['Ka'], gmpModel['parameters']['La']
    nSample = len(x)
    PPHI = np.zeros((nSample, gmpModel['nbCoef']), dtype=np.complex128)

    for l in range(La):
        it = np.concatenate((np.zeros(l), x[:nSample-l]))   # x delayed by l
        mag = np.abs(it)                      # hoisted out of the k loop
        for k in range(Ka):
            PPHI[:, (La * k) + l] = mag**k * it

    return PPHI


# ===========================================================================
# LAYER 3 -- FITTING THE COEFFICIENTS
#
# Solve y = PPHI @ c for c. The argument swap in identif_dpd_GMP is what turns
# a forward PA fit into the inverse a predistorter needs -- read that function
# before changing any call site.
# ===========================================================================

def algoLMS(PPHI, out, start, nSample):
    """
    Classical LMS Algorithm
    gmpModel['coefGMP'] = np.linalg.inv((np.dot(np.conj(PPHI.T), PPHI))) @ np.conj(PPHI.T) @ out
    """

    """
    LMS Algorithm with regularization
    """
    # Despite the name this is not an iterative LMS -- it is a one-shot
    # regularized pseudo-inverse.
    PHI = PPHI[start:nSample, :]
    U, S, V = np.linalg.svd(np.conj(PHI.T) @ PHI)
    # THE REGULARIZATION: singular values at or below 1e-8 are mapped to inf,
    # so 1/inf = 0 and those directions are discarded rather than inverted.
    # PPHI's columns are strongly correlated (|x|^6 x looks much like |x|^4 x),
    # so at high Ka the matrix goes near-singular and this truncation is the
    # only thing preventing a blow-up. It is also why (9, 9) still returns
    # something -- just something useless, -9.5 dB, worse than no DPD. Hence
    # KA_LA pinned at (7, 5) in cosimulation.py.
    Splus = np.diag(1 / np.where(S > 1e-8, S, np.inf))

    # Regularized pseudo-inverse core, formed once. The original built it twice
    # -- once for theta, once for a matrix H that nothing in this project ever
    # read. H is (nbCoef x nSample) complex128: 143 MB at nSample = 256k,
    # nbCoef = 35. Dropping it removes that matmul and its allocation, worth
    # roughly 1.2x on a fit. It does not lower peak RSS, because
    # normalize_and_sync's 16x resample has already peaked higher by then.
    pinv_core = U @ Splus @ np.conj(U.T)
    theta = pinv_core @ np.conj(PHI.T) @ out[start:nSample]

    return theta


def estimGMPopt(x, y, gmpModel):
    """Fit coefficients so that PPHI(x) @ coef best reproduces y."""
    p = gmpModel['parameters']
    gmpModel['nbCoef'] = p['Ka'] * p['La']
    nSample = len(x)
    PPHI = _build_pphi(x, gmpModel)

    gmpModel['coefGMP'] = algoLMS(PPHI, y, 0, nSample)
    gmpModel['outEst'] = PPHI @ gmpModel['coefGMP']
    return gmpModel


def identif_dpd_GMP(x, y, dpdModel):

    # THE TRAP IN THIS FILE. estimGMPopt expects (x, y) and is handed (y, x):
    # the regressors are built from the OUTPUT and fitted to predict the INPUT.
    # Followed through:
    #     get_MP_model(7, 5, paIn, paOut)
    #       -> identif_dpd_GMP(paIn, paOut)
    #         -> estimGMPopt(paOut, paIn)   regressors paOut, target paIn
    # which fits paOut -> paIn, the amplifier's inverse -- what a predistorter
    # is. Calling get_MP_model(ka, la, pa_out_n, pa_in_n) double-swaps, fits
    # the forward PA model, and yields -8 dB instead of +13 dB. The call site
    # in cosimulation.py carries a matching comment.
    dpdModel = estimGMPopt(y, x, dpdModel)
    dpdModel['nmseTimedB'] = nmse(x, dpdModel['outEst'])

    return dpdModel


def nmse(xMes, xEst):
    """Error power relative to reference power, in dB."""
    error = xMes - xEst
    nmse_dB = 10 * np.log10(np.sum(np.abs(error)**2) / np.sum(np.abs(xMes)**2))
    return nmse_dB


def get_MP_model(ka, la, x, y):
    # Layer 3's public entry point: init, then identify. Note ka then la --
    # the opposite order to the (la, ka) pairs the search below reports.
    model = _init_mp_model(ka, la)
    model = identif_dpd_GMP(x, y, model)  # identify PA/DPD coefficients
    return model


# ===========================================================================
# LAYER 4 -- USING THE FITTED MODEL
#
# simGMPopt is the predistortion itself. find_scaling_factor picks the drive
# level, and is the single most expensive stage in the file: ~50 bisection
# steps, each rebuilding the full PPHI -- 1.89 s of the 2.67 s pipeline at
# 256k samples.
# ===========================================================================

def simGMPopt(x, gmpModel):
    """Apply a fitted model to a new signal."""
    PPHI = _build_pphi(x, gmpModel)
    outEst = PPHI @ gmpModel['coefGMP']

    return outEst


def find_scaling_factor(signal, DPD_Model, desired_amplitude,
                        scale_low=1e-3, scale_high=1e3,
                        tol=5e-4, max_iter=50, verbose=True):
    """
    Scale the drive so the predistorter's peak output equals desired_amplitude,
    by log-bisection on a scalar lambda. Returns (lambda, signal*lambda, pred).

    WHY: the predistorter expands peaks, and a 7th-order polynomial diverges
    violently outside the amplitude range it was fitted on. cosimulation.py
    passes max(|pa_in_n|), the largest amplitude the model ever saw.

    Midpoints are GEOMETRIC, not arithmetic: the bracket spans six orders of
    magnitude, so the bisection has to halve log(scale).

    NOTE: `signal` shadows the scipy.signal import inside this function.
    """
    def eval_at(scale):
        x = signal * scale
        y = simGMPopt(x, DPD_Model)
        return float(np.max(np.abs(y))), y, x

    amp_low, _, _ = eval_at(scale_low)
    amp_high, _, _ = eval_at(scale_high)
    if (amp_low - desired_amplitude) * (amp_high - desired_amplitude) > 0:
        raise RuntimeError(
            f"Target not bracketed: A({scale_low})={amp_low:.6g}, "
            f"A({scale_high})={amp_high:.6g}, target={desired_amplitude:.6g}.")

    for it in range(max_iter):
        scale_mid = np.sqrt(scale_low * scale_high)
        amp_mid, y_mid, x_mid = eval_at(scale_mid)
        if verbose:
            print(f"[{it+1:02d}] scale={scale_mid:.6g}, peak={amp_mid:.6f}")
        if abs(amp_mid - desired_amplitude) < tol:
            if verbose:
                print(f"OK in {it+1} iterations.")
            return scale_mid, x_mid, y_mid
        # keep the half-interval that still brackets the target
        if (amp_low - desired_amplitude) * (amp_mid - desired_amplitude) <= 0:
            scale_high, amp_high = scale_mid, amp_mid
        else:
            scale_low, amp_low = scale_mid, amp_mid

    if verbose:
        print("Warning: max_iter reached without meeting tol.")
    return scale_mid, x_mid, y_mid


# ===========================================================================
# EXTRAS -- the PSD estimator
# ===========================================================================
def PSD_welch(x, fs, win):
    """
    Power spectral density, returned as (pxx, freq) with DC in the middle.

    return_onesided=False because x is a complex baseband envelope: negative
    frequencies are real and distinct. The window is the length of the whole
    record, so nperseg = len(x) and Welch averages exactly ONE segment -- this
    is a single windowed periodogram, best resolution but no variance
    reduction.
    """
    beta = 8.6
    if win == 'kaiser':
        window = get_window(('kaiser', beta), len(x))
    else:
        window = get_window(win, len(x))

    freq, pxx = welch(x, fs=fs, window=window, return_onesided=False, scaling='density')

    return np.fft.fftshift(pxx), np.fft.fftshift(freq)


# ===========================================================================
# ADDED FOR THIS PROJECT -- not part of the XLIM / Rapid.Space extraction.
# ===========================================================================
def shortlist_orders(paIn, paOut, grid=range(2, 12), fit_frac=0.70,
                     tol_db=0.25, verbose=True):
    """
    Shortlist the model orders worth testing. Returns a LIST, not a winner.

    Each candidate is fitted on the first 70% of the record and scored on the
    last 30%. Scoring on held-out data is the point: a model scored on the
    samples it was fitted to cannot show overfitting, because overfitting
    improves that score.

    Returns (ka, la, n_coef, held_out_nmse_db) tuples, best score first.
    On the reference dataset 13 of the 100 grid points survive:

        ( 9, 3)  27 coef  -48.22 dB      ( 7, 4)  28 coef  -43.70 dB
        (10, 3)  30 coef  -48.19 dB      ( 7, 5)  35 coef  -43.65 dB
        ( 8, 3)  24 coef  -48.13 dB      ( 7,10)  70 coef  -41.36 dB
        ( 7, 3)  21 coef  -48.05 dB      ( 6, 8)  48 coef  -40.36 dB
        ( 9, 2)  18 coef  -47.74 dB      ( 6, 7)  42 coef  -39.84 dB
        ( 7, 6)  42 coef  -43.73 dB      ( 6, 9)  54 coef  -39.59 dB
                                         ( 6,11)  66 coef  -38.07 dB

    We ran all of these through the ADS loop manually. (7, 5) gave the best
    result, on both NMSE and ACPR, and that is what cosimulation.py pins.
    The ordering above is only a shortlist -- it does not predict the winner.
    """
    cut = int(fit_frac * len(paIn))
    in_fit, out_fit = paIn[:cut], paOut[:cut]
    in_val, out_val = paIn[cut:], paOut[cut:]

    scored = []
    for la in grid:
        for ka in grid:
            model = get_MP_model(ka, la, in_fit, out_fit)
            predicted = simGMPopt(out_val, model)
            scored.append((ka, la, ka * la, nmse(in_val, predicted)))

    best = min(s[3] for s in scored)
    tied = {s[:2] for s in scored if s[3] <= best + tol_db}
    per_depth = {min((s for s in scored if s[1] == la), key=lambda r: r[3])[:2]
                 for la in grid}
    keep = tied | per_depth
    shortlist = sorted((s for s in scored if s[:2] in keep), key=lambda r: r[3])

    if verbose:
        print(f"  shortlist_orders: {len(scored)} candidates, fit on {cut}, "
              f"scored on {len(paIn) - cut} held out")
        print(f"    best held-out {best:.2f} dB; {len(tied)} within {tol_db} dB, "
              f"plus the best at each of {len(per_depth)} memory depths")
        for ka, la, n_coef, val in shortlist:
            print(f"      ka={ka:2d} la={la:2d}  {n_coef:3d} coef  {val:7.2f} dB")
        print("    -> a shortlist, not a winner. Closed-loop testing chose (7,5);"
              " see the docstring.")
    return shortlist
