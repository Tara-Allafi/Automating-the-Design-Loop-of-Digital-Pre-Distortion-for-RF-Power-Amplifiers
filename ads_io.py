"""
ads_io.py -- everything that talks to ADS, shared by amam.py and cosimulation.py.

Open a workspace, edit a schematic in memory, netlist it, simulate, read the
results back. Plus the signal file I/O, since a stimulus file is how a signal
gets into ADS in the first place.
"""
import shutil
import time
from pathlib import Path


import numpy as np
import pandas as pd

from keysight.ads import de
from keysight.edatoolbox import ads
import keysight.ads.dataset as dataset


# ===========================================================================
# CONFIG
# ===========================================================================
WORKSPACE = Path(__file__).resolve().parent / "ADS_workspace"
DATA = WORKSPACE / "data"        # ADS resolves modfile() names against this
LIB = "DPA_simulation_lib"

FUND = 2.0e9                     # carrier, PORT1 Freq[1]
SR = 122.88e6                    # sample rate, VAR1 SR
R = 50.0                         # port impedance
BW = 5e6                         # LTE carrier bandwidth

AMAM_CELL = "DPA_AM_AM"          # harmonic-balance Pin sweep
MOD_CELL = "DPA_Modulated_TB"    # envelope testbench

# Both the reference and the linearized run use MOD_CELL -- only the stimulus
# file changes. Checked against the netlists: DPD_TB is the same amplifier,
# bias and topology, differing only in auto-generated node names, and
# DPA_Modulated_TB additionally carries a P_Probe on its input. One schematic
# does both jobs.
STIMULUS = "LTE_DL_7p68MSPS_5M_SR_122.88.txt"


# ===========================================================================
# ADS
# ===========================================================================
def open_workspace():
    return de.open_workspace(str(WORKSPACE))


def close_workspace():
    de.close_workspace()


def open_design(ws, cell):
    """READ_ONLY: edits live in memory and reach the netlist, disk is untouched."""
    return ws.design((LIB, cell, "schematic"))


def set_vars(design, values, inst="VAR1"):
    """Write the VarEqn box: Pin, SR, nbrpt, F."""
    for key, val in values.items():
        design.instances[inst].vars[key] = str(val)
    design.instances[inst].update_item_annotation()


def set_modfile(design, filename, inst="PORT1"):
    """Point the modulated source at a stimulus file sitting in DATA."""
    design.instances[inst].parameters["Mod"].value = \
        f'modfile(time,"{Path(filename).name}","tri",SR)'


def simulate(design, dataset_name):
    """
    Netlist and run.

    hpeesofsim writes <dataset_name>.ds into the directory it RUNS FROM, and
    the workspace root is the only place lib.defs resolves the MA_RFP PDK.
    So: run there, then move the result into data/.
    """
    t0 = time.time()
    ads.CircuitSimulator().run_netlist(
        design.generate_netlist(),
        output_dir=str(DATA),
        working_dir=str(WORKSPACE),
        rel_data_dir=str(DATA),      # where modfile() looks for its .txt
        dataset_name=dataset_name,
    )
    # Generates the netlist and runs the simulation. Results land in working_dir
    # rather than output_dir, so move them back into the ADS workspace data folder,
    # where ADS expects the dataset and modfile() expects its .txt stimulus file.

    for name in (f"{dataset_name}.ds", f"{dataset_name}_data"):
        src = WORKSPACE / name
        if src.exists():
            dst = DATA / name
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            src.replace(dst)
    print(f"      simulated in {time.time() - t0:.1f}s")
    return DATA / f"{dataset_name}.ds"


def extract(dataset_name, axis="time"):
    """
    Dataset -> (axis, complex Vin, complex Vout), sliced at the fundamental.

    axis is "time" for an envelope run, "Pin" for the harmonic-balance sweep.
    Both testbenches name their nets Vin / Vout.
    """
    ds = dataset.open(DATA / f"{dataset_name}.ds")
    vb = next(n for n in ds.varblock_names
              if "Vin" in ds[n].to_dataframe().columns)
    df = ds[vb].to_dataframe().reset_index()
    df = df[np.isclose(df["freq"], FUND)].sort_values(axis)
    return (df[axis].to_numpy(float),
            df["Vin"].to_numpy(), df["Vout"].to_numpy())


# ===========================================================================
# SIGNALS
# ===========================================================================
def dbm(v):
    """Real power in dBm. P = |V|^2/(2R); the 2R because ADS reports peak."""
    power_w = np.abs(np.asarray(v)) ** 2 / (2 * R)
    power_dbm = 30 + 10 * np.log10(power_w + 1e-30)
    return power_dbm


def mean_dbm(v):
    """Average power in dBm -- average the power, THEN take the log."""
    mean_power_w = np.mean(np.abs(v) ** 2) / (2 * R)
    power_dbm = 30 + 10 * np.log10(mean_power_w + 1e-30)
    return power_dbm


def write_tri(sig, filename):
    """Write a complex envelope as the headerless TRI file ADS reads."""
    path = DATA / Path(filename).name
    pd.DataFrame({"Time": np.arange(sig.size) / SR,
                  "Real": sig.real,
                  "Imaginary": sig.imag}).to_csv(path, index=False, header=False)
    return path.name


def read_tri(path, n=None):
    df = pd.read_csv(path, names=["t", "re", "im"], nrows=n)
    complex_signal = (df["re"] + 1j * df["im"]).to_numpy()
    return complex_signal

# index start from 0 in python nbrpt max index needs to be minus 1 from the number of samples.
def envelope_vars(n_samples, pin_dbm):
    """nbrpt counts STEPS, so a 6400-sample run carries nbrpt=6399."""
    nbrpt = n_samples - 1                  # ADS VAR1.nbrpt: steps, not samples
    sample_rate_expr = f"{SR/1e6:g}*1e6"   # matches VAR1.SR's own expression style
    return {"nbrpt": nbrpt, "SR": sample_rate_expr, "Pin": pin_dbm}
