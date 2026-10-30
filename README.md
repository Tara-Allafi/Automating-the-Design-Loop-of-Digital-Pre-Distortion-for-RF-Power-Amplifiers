# ADS ⇄ Python DPD co-simulation

Closes the loop between a Keysight ADS circuit simulation and a digital
predistortion algorithm, with no manual step in between:

```
ADS reference run ──▶ identify DPD ──▶ inject predistorted signal ──▶ ADS run ──▶ score
        ▲                                                                    │
        └────────────────────────────────────────────────────────────────────┘
```

Self-contained — nothing outside this folder is imported, and the only external
things it touches are the ADS install and the workspace.

## Run it

Two independent operations:

```bash
./adsenv.sh amam.py          # static AM/AM sweep         (~5 s, 2 figures)

./adsenv.sh cosimulation.py  # the modulated DPD loop     (~20 s)
./adsenv.sh plot.py          # its eight figures          (~15 s, opens them)
```

`amam.py` stands alone — one CW tone stepped in power, no signal file, no DPD.
`cosimulation.py` then `plot.py` are the modulated loop; `plot.py` reads what
`cosimulation.py` wrote, so run them in that order.

Always through `adsenv.sh`, never `python3` directly: the ADS bindings are
compiled extensions that need `LD_LIBRARY_PATH` set at process launch. The
wrapper scopes it to that one process — exporting it into your shell would
shadow the system OpenSSL and break `curl` and the ADS GUI.

## Setup on a new machine

No path in this repo is tied to one machine. You need:

1. **Keysight ADS** (developed on 2026 Update 2) with a license. `adsenv.sh`
   finds it from `$HPEESOF_DIR` or the `ads` command on your PATH; otherwise
   set `ADS_HOME=/path/to/ADS`.
2. **The `MA_RFP_ADS_v1p8` design kit**, copied to
   `ADS_workspace/MA_RFP_ADS_v1p8/`. It is licensed, so it is not in the
   repo.
3. **The Python environment and the LTE stimulus**, set up once:

   ```bash
   ./adsenv.sh --setup    # .venv from ADS's own Python, plus torch
   ```

   This also unpacks the stimulus `LTE_DL_7p68MSPS_5M_SR_122.88.txt` (67 MB),
   committed xz-compressed in `ADS_workspace/data/`.

   It reuses the keysight bindings, numpy, scipy, pandas and matplotlib that
   ADS ships, and adds only what `requirements.txt` lists. To use an existing
   interpreter instead, set `ADS_PY=/path/to/python3`.

The workspace is found relative to this folder, so the repo can be cloned
anywhere.

## Files

| file | what it is |
|---|---|
| `ads_io.py` | everything that talks to ADS: open, edit, netlist, simulate, read back, plus signal file I/O |
| `dpd_modeling_MP.py` | the DPD algorithms — memory-polynomial fitting, normalization, sync, scaling search |
| `amam.py` | static AM/AM sweep — a separate operation, its own figures |
| `cosimulation.py` | the modulated DPD loop |
| `plot.py` | the loop's eight figures + ACPR |
| `dpd_lstm_inverse.py` | the LSTM predistorter: preprocessing, model, training, all in one file |
| `compare_dpd.py` | memory polynomial vs LSTM, both pushed back through ADS |
| `adsenv.sh` | the only launcher; run any script in this folder through it |

## One schematic, two runs

Both the reference and the linearized run use **`DPA_Modulated_TB`**. Only the
stimulus file changes — the plain LTE waveform, then the predistorted one.

`DPD_TB` is not used. Compared netlist to netlist it is the same amplifier
(`DPA_design:X3`), the same bias (28 V, −5 V, −3 V), the same shorts and the
same envelope controller; it differs only in auto-generated internal node names
and in *lacking* the input-power probe `DPA_Modulated_TB` carries. One
schematic does both jobs, and both runs then report their nets under the same
names, which removes the per-cell name registry the code used to need.

`dpd_modeling_MP.py` was extracted **verbatim** from the XLIM / Rapid.Space
`DPD_modeling` code — pulled function by function with an AST parser so the
numerics are byte-identical — keeping only what the loop calls. Left behind:
the PA-modelling path, CCDF, plotting helpers, and the import-time matplotlib
side effects that previously forced a working-directory change and a monkey-
patched `savefig` just to import the file.

## Results

```
distortion vs wanted signal : -27.50 dB  ->  -40.87 dB   (+13.37 dB)
ACPR                        : -44.62 dB  ->  -63.54 dB   (+18.92 dB)
mean output power           :  27.56 dBm ->   27.54 dBm
```

From the AM/AM sweep: small-signal gain 16.49 dB, 5.73 dB of compression across
the full sweep, and **1 dB compression at Pin = 12.24 dBm**. The modulated runs
drive the amplifier at 12.19 dBm average — essentially exactly at P1dB, which
is where DPD has to earn its keep.

Distortion power cut to ~1/22nd at unchanged output power. The unchanged power
matters: linearity bought by backing off would be no gain at all.

Consistency checks that the numbers are real:

| check | result |
|---|---|
| measured drive vs `VAR1.Pin` | 12.19 dBm vs 12.19 — exact |
| both runs at same drive | `rms(Vin)` = `rms(Vin_dpd)` = 1.2868 |
| gain vs the independent HB sweep | 15.37 dB mean, 16.49 dB small-signal → 1.64 dB compression |
| PAPR in → out, no DPD | 8.85 → 8.34 dB (peaks squashed) |
| PAPR in → out, with DPD | 8.85 → 8.83 dB (peaks restored) |

That last pair is the mechanism in four numbers: compression eats 0.51 dB of
peak-to-average, the DPD adds 0.54 dB back before the amplifier, and the output
lands within 0.02 dB of the input's PAPR.

## Things that will bite you

**The model order is pinned, not searched.** `KA_LA = (7, 5)` in
`cosimulation.py`. The automatic search minimizes fit error on the training
pair, which rewards the highest order on the grid and lands on an
ill-conditioned inverse — measured against one fixed reference, `(9,9)` gives
**−9.5 dB, worse than no DPD**, while `(3,2)`…`(11,7)` all give +12…+13.4 dB.
Set `KA_LA = None` to use the search, but validate what it picks.

**Argument order decides what you fit.** `get_MP_model(ka, la, paIn, paOut)`
fits the amplifier's *inverse* — a DPD. Swap the last two and you fit the
forward PA model, which predistorts with the amplifier's own nonlinearity and
roughly doubles the distortion. Nothing but position distinguishes them.

**The dataset lands in the working directory, not `output_dir`.**
`run_netlist(output_dir=…)` only supplies a default working dir; hpeesofsim
writes `<name>.ds` into its CWD. The workspace root is also the only CWD from
which `lib.defs` resolves the `MA_RFP_ADS_v1p8` PDK, so runs happen there and
`simulate()` moves the result into `data/` afterwards.

**Failed simulations are silent.** `run_netlist()` returns `None` on a fatal
netlist error, leaving a stale dataset that reads back as plausible numbers.
This version does not check — if results look wrong, read the ADS log.

**ADS envelope runs are not bit-reproducible.** Two identical runs differ by up
to 3.3e-4 in `Vout`. Harmless at this scale, but don't quote ACPR to two
decimals as exact.

**`nbrpt` counts steps, not samples.** A 6400-sample run carries `nbrpt=6399`.
`envelope_vars()` derives it so a staged signal and the run length cannot drift.

**Editing is in-memory only.** Designs open `READ_ONLY`; edits reach the
netlist but never the disk, so schematics cannot be corrupted and the ADS GUI
can stay open. Nothing here ever calls `save_design()`.

## The plots

Section 1 is without DPD, section 2 with. AM/AM plots are in real dBm; the
"normalized" ones subtract the small-signal gain so a perfectly linear PA lands
on `y = x`. Spectra marked "in-band aligned" have the gain removed so the flat
tops coincide and the shoulders compare directly — ACPR is a ratio, so that
scaling changes what you see, not what is measured, and the aligned and
unaligned ACPR numbers are identical by construction.
