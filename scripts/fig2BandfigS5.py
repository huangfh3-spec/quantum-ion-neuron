#!/usr/bin/env python
# coding: utf-8
"""

Stokes reconstruction:
    Sx = P_dark(xpositive) - P_dark(xminus)
    Sy = P_dark(ypositive) - P_dark(yminus)

Memristor variables:
    Iq(t)   = -0.75 * Sy(t)
    Vq(t_i) = [Sy(t_{i+1}) - Sy(t_{i-1})] / (2 dt) - delta * Sx(t_i)
"""

from __future__ import annotations

import ast
import itertools
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
from qutip import Options, basis, mesolve

warnings.filterwarnings(
    "ignore",
    message="Dedicated options class are no longer needed.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="e_ops will be keyword only.*",
    category=FutureWarning,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "Fig_data"
if not DATA.is_dir():
    raise FileNotFoundError(f"Figure data directory not found: {DATA}")


# ------------------------ User-facing parameters ------------------------

DIM = 7
TOTAL_T = 160
DT = STEP = 4
DELTA = 0.040
GAMMA = 19.6 * 2 * np.pi
MEASURE_TIMES = int(TOTAL_T // DT + 1)

DARK_THRESHOLD = 2
FFT_CUTOFF_RATIO = 0.35  # main-figure default; keep aligned with Fig. 2B if needed

# Cutoff robustness table settings. These are matched to the LaTeX table in the manuscript:
# nominal cutoff = 0.35 f_max; tested cutoffs = 0.25--0.45 f_max.
ROBUSTNESS_NOMINAL_RATIO = 0.35
ROBUSTNESS_RATIOS = (0.25, 0.30, 0.35, 0.40, 0.45)
N_GROUP_BOOT = 5**5
# N_SHOT_BOOT = 2000
N_BOOT_PLOT = 5
RNG_SEED = 20260515

STOKES_DIR = DATA
STOKES_BASENAME = "Stokes_memristor"
N_STOKES_GROUPS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENTARY_OUTPUT_DIR = ROOT / "figures" / "supplementary"
SUPPLEMENTARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCIENCE_TEMPLATE_DIR = ROOT / "science_template"

QUTIP_OPTIONS = Options(nsteps=1500000)


# ------------------------ Stokes CSV loading ------------------------


def find_stokes_file(index: int, stokes_dir: Path = STOKES_DIR) -> Path:
    """Find Stokes_memristor_i in the data folder, allowing common extensions."""
    base = f"{STOKES_BASENAME}_{index}"
    candidates = [
        stokes_dir / base,
        stokes_dir / f"{base}.csv",
        stokes_dir / f"{base}.txt",
        stokes_dir / f"{base}.dat",
    ]
    candidates.extend(sorted(stokes_dir.glob(f"{base}.*")))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    tried = ", ".join(str(c) for c in candidates[:4])
    raise FileNotFoundError(f"Cannot find {base} in {stokes_dir}. Tried: {tried}")


def read_stokes_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one Stokes file with columns time, Sx, Sy.

    The function accepts comma-, tab-, or whitespace-separated files. A header row
    is allowed; non-numeric rows are dropped after conversion.
    """
    try:
        df = pd.read_csv(path, header=None, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(path, header=None, delim_whitespace=True, comment="#")
    if df.shape[1] < 3:
        raise ValueError(f"{path} must contain at least three columns: time, Sx, Sy.")

    cols = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    cols = cols.dropna(how="any")
    if cols.empty:
        raise ValueError(f"{path} contains no numeric rows in the first three columns.")

    time = cols.iloc[:, 0].to_numpy(dtype=float)
    sx = cols.iloc[:, 1].to_numpy(dtype=float)
    sy = cols.iloc[:, 2].to_numpy(dtype=float)
    return time, sx, sy


def load_stokes_replicates(
    stokes_dir: Path = STOKES_DIR,
    n_groups: int = N_STOKES_GROUPS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load five Stokes_memristor_i files and return Sx/Sy replicate arrays.

    Returns
    -------
    sx, sy : ndarray
        Arrays of shape (n_groups, n_time).
    labels : list[str]
        Labels Stokes_memristor_1 ... Stokes_memristor_5.
    """
    sx_reps: list[np.ndarray] = []
    sy_reps: list[np.ndarray] = []
    time_reps: list[np.ndarray] = []
    labels: list[str] = []

    for i in range(1, n_groups + 1):
        path = find_stokes_file(i, stokes_dir)
        time, sx, sy = read_stokes_csv(path)
        time_reps.append(time)
        sx_reps.append(sx)
        sy_reps.append(sy)
        labels.append(f"Stokes_memristor_{i}")
        print(f"Loaded {labels[-1]} from {path} with {len(time)} time points.")

    common_len = min([len(arr) for arr in sx_reps + sy_reps + time_reps])
    if common_len < 3:
        raise ValueError("Need at least three common time points for center-difference Vq reconstruction.")

    # Check time grids. The downstream reconstruction still uses STEP=4 us, so
    # warn if the imported time column is not consistent with it.
    ref_time = time_reps[0][:common_len]
    expected_time = np.arange(common_len, dtype=float) * STEP
    if not np.allclose(ref_time, expected_time, atol=1e-9, rtol=1e-9):
        print(
            "Warning: time column in Stokes_memristor_1 is not exactly "
            f"0, {STEP}, 2*{STEP}, ... us. Downstream reconstruction still uses STEP={STEP} us."
        )
    for idx, time in enumerate(time_reps[1:], start=2):
        if not np.allclose(time[:common_len], ref_time, atol=1e-9, rtol=1e-9):
            print(f"Warning: time grid of Stokes_memristor_{idx} differs from Stokes_memristor_1.")

    sx_arr = np.vstack([arr[:common_len] for arr in sx_reps])
    sy_arr = np.vstack([arr[:common_len] for arr in sy_reps])
    return sx_arr, sy_arr, labels


# ------------------------ Iq-Vq reconstruction and uncertainty ------------------------


def reconstruct_iq_vq_replicates(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    sample_step: float = STEP,
    delta: float = DELTA,
) -> dict[str, np.ndarray]:
    """Reconstruct Iq and Vq for every repeated group."""
    sx_reps = np.asarray(sx_reps, dtype=float)
    sy_reps = np.asarray(sy_reps, dtype=float)
    if sx_reps.shape != sy_reps.shape:
        raise ValueError(f"Sx and Sy replicate arrays must match. Got {sx_reps.shape} and {sy_reps.shape}.")
    if sx_reps.shape[1] < 3:
        raise ValueError("Need at least 3 time points for center-difference Vq reconstruction.")

    t_full = np.arange(sx_reps.shape[1], dtype=float) * sample_step
    t_mid = t_full[1:-1]
    i_rep = -0.75 * sy_reps[:, 1:-1]
    v_rep = (sy_reps[:, 2:] - sy_reps[:, :-2]) / (2 * sample_step) - delta * sx_reps[:, 1:-1]
    if i_rep.shape[0] > 1:
        i_sem = np.std(i_rep, axis=0, ddof=1) / np.sqrt(i_rep.shape[0])
        v_sem = np.std(v_rep, axis=0, ddof=1) / np.sqrt(v_rep.shape[0])
    else:
        i_sem = np.zeros(i_rep.shape[1], dtype=float)
        v_sem = np.zeros(v_rep.shape[1], dtype=float)

    return {
        "t": t_full,
        "t_mid": t_mid,
        "I_rep": i_rep,
        "V_rep": v_rep,
        "I_mean": np.mean(i_rep, axis=0),
        "V_mean": np.mean(v_rep, axis=0),
        "I_sem": i_sem,
        "V_sem": v_sem,
    }


def close_loop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append the first point to close a parametric loop."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.r_[x, x[0]], np.r_[y, y[0]]


def loop_area(iq: np.ndarray, vq: np.ndarray) -> float:
    """Signed hysteresis loop area, A = integral Vq dIq."""
    iq_closed, vq_closed = close_loop(iq, vq)
    return float(np.trapz(vq_closed, iq_closed))


# def group_bootstrap_loops(
#     i_rep: np.ndarray,
#     v_rep: np.ndarray,
#     n_boot: int = N_GROUP_BOOT,
#     seed: int = RNG_SEED,
# ) -> dict[str, np.ndarray]:
#     """Bootstrap mean Iq-Vq trajectories by resampling complete repeated groups."""
#     rng = np.random.default_rng(seed)
#     i_rep = np.asarray(i_rep, dtype=float)
#     v_rep = np.asarray(v_rep, dtype=float)
#     if i_rep.shape != v_rep.shape:
#         raise ValueError("I and V replicate arrays must have the same shape.")

#     n_groups, n_points = i_rep.shape
#     boot_i = np.empty((n_boot, n_points), dtype=float)
#     boot_v = np.empty((n_boot, n_points), dtype=float)
#     boot_area = np.empty(n_boot, dtype=float)

#     for k in range(n_boot):
#         idx = rng.integers(0, n_groups, size=n_groups)
#         boot_i[k] = np.mean(i_rep[idx], axis=0)
#         boot_v[k] = np.mean(v_rep[idx], axis=0)
#         boot_area[k] = loop_area(boot_i[k], boot_v[k])

#     return {"I": boot_i, "V": boot_v, "area": boot_area}


def fft_lowpass(signal: np.ndarray, cutoff: float, sample_step: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=sample_step)
    mask = np.abs(freq) < cutoff
    return np.fft.ifft(fft * mask).real


def spectrum_fmax(n_points: int, sample_step: float = STEP) -> float:
    
    if n_points < 2:
        raise ValueError("Need at least two points to determine an FFT frequency grid.")
    return float(np.max(np.fft.fftfreq(n_points, d=sample_step)))


def cutoff_from_ratio(
    cutoff_ratio: float = FFT_CUTOFF_RATIO,
    sample_step: float = STEP,
    n_points: int | None = None,
) -> float:
   
    if n_points is None:
        raise ValueError("n_points is required because f_max depends on FFT length.")
    return cutoff_ratio * spectrum_fmax(n_points, sample_step)


def reconstruct_mean_then_fft_loop(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    sample_step: float = STEP,
    delta: float = DELTA,
    cutoff_ratio: float = FFT_CUTOFF_RATIO,
    apply_fft: bool = True,
) -> dict[str, np.ndarray]:
    """Average Sx/Sy across repeated groups, reconstruct Iq/Vq, then FFT filter.

    This follows the requested display convention for the main figure:
    1. reconstruct Sx/Sy for each complete experimental group,
    2. average the five groups in Sx/Sy space,
    3. compute raw Iq and Vq from the averaged trajectory,
    4. apply the same FFT low-pass filter to Iq and Vq.
    """
    sx_reps = np.asarray(sx_reps, dtype=float)
    sy_reps = np.asarray(sy_reps, dtype=float)
    if sx_reps.shape != sy_reps.shape:
        raise ValueError(f"Sx and Sy replicate arrays must match. Got {sx_reps.shape} and {sy_reps.shape}.")
    if sx_reps.shape[1] < 3:
        raise ValueError("Need at least 3 time points for center-difference Vq reconstruction.")

    sx_mean = np.mean(sx_reps, axis=0)
    sy_mean = np.mean(sy_reps, axis=0)
    t_full = np.arange(sx_mean.shape[0], dtype=float) * sample_step
    t_mid = t_full[1:-1]
    i_raw = -0.75 * sy_mean[1:-1]
    v_raw = (sy_mean[2:] - sy_mean[:-2]) / (2 * sample_step) - delta * sx_mean[1:-1]
    cutoff = cutoff_from_ratio(cutoff_ratio, sample_step, len(v_raw))
    if apply_fft:
        i_fft = fft_lowpass(i_raw, cutoff, sample_step)
        v_fft = fft_lowpass(v_raw, cutoff, sample_step)
    else:
        i_fft = i_raw.copy()
        v_fft = v_raw.copy()

    return {
        "t": t_full,
        "t_mid": t_mid,
        "Sx_mean": sx_mean,
        "Sy_mean": sy_mean,
        "I_raw": i_raw,
        "V_raw": v_raw,
        "I_fft": i_fft,
        "V_fft": v_fft,
        "f_max": spectrum_fmax(len(v_raw), sample_step),
        "cutoff": cutoff,
        "cutoff_ratio": cutoff_ratio,
    }


def group_bootstrap_mean_then_fft(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    sample_step: float = STEP,
    delta: float = DELTA,
    cutoff_ratio: float = FFT_CUTOFF_RATIO,
    apply_fft: bool = True,
) -> dict[str, np.ndarray]:
    """Exhaustively enumerate all group-level bootstrap resamples."""
    sx_reps = np.asarray(sx_reps, dtype=float)
    sy_reps = np.asarray(sy_reps, dtype=float)
    if sx_reps.shape != sy_reps.shape:
        raise ValueError("Sx and Sy replicate arrays must have the same shape.")

    n_groups = sx_reps.shape[0]
    n_points = sx_reps.shape[1] - 2
    n_boot = n_groups**n_groups
    boot_i = np.empty((n_boot, n_points), dtype=float)
    boot_v = np.empty((n_boot, n_points), dtype=float)
    boot_area = np.empty(n_boot, dtype=float)

    for k, idx_tuple in enumerate(itertools.product(range(n_groups), repeat=n_groups)):
        idx = np.asarray(idx_tuple, dtype=int)
        rec = reconstruct_mean_then_fft_loop(
            sx_reps[idx],
            sy_reps[idx],
            sample_step=sample_step,
            delta=delta,
            cutoff_ratio=cutoff_ratio,
            apply_fft=apply_fft,
        )
        boot_i[k] = rec["I_fft"]
        boot_v[k] = rec["V_fft"]
        boot_area[k] = loop_area(boot_i[k], boot_v[k])

    return {"I": boot_i, "V": boot_v, "area": boot_area, "n_resamples": np.asarray(n_boot)}


def smooth_bootstrap_vq(
    v_boot: np.ndarray,
    cutoff_ratio: float = FFT_CUTOFF_RATIO,
    sample_step: float = STEP,
) -> np.ndarray:
    """Low-pass smooth bootstrap Vq trajectories for Fig. S5 visualization only."""
    v_boot = np.asarray(v_boot, dtype=float)
    cutoff = cutoff_from_ratio(cutoff_ratio, sample_step, v_boot.shape[1])
    return np.vstack([fft_lowpass(row, cutoff, sample_step) for row in v_boot])


def old_pointwise_vq_sem(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    sample_step: float = STEP,
    delta: float = DELTA,
) -> np.ndarray:
    """Reproduce the old pointwise SEM style used in fig2BandfigS5.py."""
    sx_sem = np.std(sx_reps, axis=0, ddof=1) / np.sqrt(sx_reps.shape[0])
    sy_sem = np.std(sy_reps, axis=0, ddof=1) / np.sqrt(sy_reps.shape[0])
    dy_dt_sem = np.sqrt(sy_sem[2:] ** 2 + sy_sem[:-2] ** 2) / (2 * sample_step)
    return 0.5 * np.sqrt(dy_dt_sem**2 + (delta * sx_sem[1:-1]) ** 2)


def validate_against_mean_csv(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    tolerance: float = 5e-4,
) -> dict[str, float]:
    """Optionally compare the five-group mean with sx_mean.csv/sy_mean.csv.

    This check is diagnostic only in the Stokes-CSV workflow: the five input
    Stokes files are treated as the source of truth. If the mean CSV files are
    absent or differ slightly, the script continues and records the diagnostics.
    """
    sx_path = DATA / "sx_mean.csv"
    sy_path = DATA / "sy_mean.csv"
    if not sx_path.exists() or not sy_path.exists():
        return {
            "sx_max_abs": float("nan"),
            "sy_max_abs": float("nan"),
            "sx_rmse": float("nan"),
            "sy_rmse": float("nan"),
        }

    sx_csv = pd.read_csv(sx_path, header=None).iloc[:, 0].to_numpy(dtype=float)
    sy_csv = pd.read_csv(sy_path, header=None).iloc[:, 0].to_numpy(dtype=float)
    sx_mean = np.mean(sx_reps, axis=0)
    sy_mean = np.mean(sy_reps, axis=0)

    n_sx = min(len(sx_csv), len(sx_mean))
    n_sy = min(len(sy_csv), len(sy_mean))
    sx_max_abs = float(np.max(np.abs(sx_mean[:n_sx] - sx_csv[:n_sx])))
    sy_max_abs = float(np.max(np.abs(sy_mean[:n_sy] - sy_csv[:n_sy])))
    result = {
        "sx_max_abs": sx_max_abs,
        "sy_max_abs": sy_max_abs,
        "sx_rmse": float(np.sqrt(np.mean((sx_mean[:n_sx] - sx_csv[:n_sx]) ** 2))),
        "sy_rmse": float(np.sqrt(np.mean((sy_mean[:n_sy] - sy_csv[:n_sy]) ** 2))),
    }
    if sx_max_abs > tolerance or sy_max_abs > tolerance:
        print(
            "Warning: Stokes CSV group mean differs from sx_mean.csv/sy_mean.csv "
            f"beyond {tolerance}. Diagnostics: {result}"
        )
    return result


# ------------------------ Simulation ------------------------


def run_qutip_simulation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the original seven-level simulation and return t_mid, Iq, Vq."""
    basis_states = [basis(DIM, i) for i in range(DIM)]

    sigx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
    sigy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
    sigz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
    identity_qubit = basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
    identity_excited = sum(basis_states[j] * basis_states[j].dag() for j in range(2, DIM))

    def microwave_rotation(theta: float, phi: float):
        return (
            np.cos(theta / 2) * identity_qubit
            - 1j * np.sin(theta / 2) * (np.cos(phi) * sigx + np.sin(phi) * sigy)
            + identity_excited
        )

    h_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()
    collapse_ops = [
        *[np.sqrt(GAMMA / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
        *[np.sqrt(GAMMA / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
        *[np.sqrt(GAMMA / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]],
    ]

    def build_hamiltonian(j1_val: float):
        return j1_val * h_structure1

    dt_internal = np.linspace(0, DT, 80)
    states = (basis_states[1] + basis_states[0]).unit()
    j1 = [0.0]
    gamma_real = [0.0]
    sx = [1.0]
    sy = [0.0]

    for i in range(MEASURE_TIMES - 1):
        h2 = build_hamiltonian(j1[i])
        result = mesolve(
            h2,
            states,
            dt_internal,
            collapse_ops,
            [
                sigx,
                sigy,
                sigz,
                basis_states[0] * basis_states[0].dag(),
                basis_states[1] * basis_states[1].dag(),
                basis_states[2] * basis_states[2].dag(),
                basis_states[3] * basis_states[3].dag(),
            ],
            options=QUTIP_OPTIONS,
        )
        state_result = mesolve(h2, states, dt_internal, collapse_ops, [], options=QUTIP_OPTIONS)

        rho_y = (
            microwave_rotation(np.pi / 2, -DELTA * DT * (i + 1))
            * state_result.states[-1]
            * microwave_rotation(np.pi / 2, -DELTA * DT * (i + 1)).dag()
        )
        rho_x = (
            microwave_rotation(np.pi / 2, -DELTA * DT * (i + 1) - np.pi / 2)
            * state_result.states[-1]
            * microwave_rotation(np.pi / 2, -DELTA * DT * (i + 1) - np.pi / 2).dag()
        )

        sx.append(float(np.real(rho_x[1, 1] - rho_x[0, 0])))
        sy.append(float(np.real(rho_y[1, 1] - rho_y[0, 0])))

        gamma_real.append(gamma_real[i] + 0.5 * sy[i] * DT * 0.0004)
        gamma_for_j = max(gamma_real[i + 1], 0.0)
        j1.append(float(np.sqrt(3 * GAMMA * gamma_for_j / 8)))
        states = state_result.states[-1]

    sx_arr = np.asarray(sx[: TOTAL_T // DT], dtype=float)
    sy_arr = np.asarray(sy[: TOTAL_T // DT], dtype=float)
    rec = reconstruct_iq_vq_replicates(sx_arr[None, :], sy_arr[None, :])
    return rec["t_mid"], rec["I_mean"], rec["V_mean"]


# ------------------------ Plotting and exports ------------------------


def set_science_style() -> None:
    plt.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.family"] = "Arial"
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 13,
            "figure.titlesize": 20,
            "mathtext.fontset": "custom",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.rm": "Arial",
            "axes.linewidth": 1.8,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axes(ax) -> None:
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.18, linestyle="--", linewidth=0.5)
    ax.set_facecolor("#f8f9fa")
    ax.tick_params(axis="both", which="major", length=6, width=1.6, color="black")
    ax.tick_params(axis="both", which="minor", length=3.5, width=1.0, color="black")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_color("black")


def style_legend(ax, fontsize: float, loc: str = "best"):
    """Apply a compact Arial legend style."""
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="0.75",
        loc=loc,
        prop={"family": "Arial", "size": fontsize},
        handlelength=1.45,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.28,
        borderaxespad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_fontname("Arial")
    return legend


def apply_arial_to_figure(fig: plt.Figure) -> None:
    """Force every Matplotlib text artist in the figure to use Arial."""
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily("Arial")
        text.set_fontname("Arial")


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure, using a fallback name if the target is open or locked."""
    try:
        fig.savefig(path, bbox_inches="tight")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(fallback, bbox_inches="tight")
        print(f"Warning: {path} is locked. Saved fallback file to {fallback}.")
        return fallback


def copy_figure(src: Path, dst: Path) -> Path:
    """Copy a generated figure to the manuscript figure folder."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return dst
    except PermissionError:
        fallback = dst.with_name(f"{dst.stem}_new{dst.suffix}")
        shutil.copy2(src, fallback)
        print(f"Warning: {dst} is locked. Copied fallback file to {fallback}.")
        return fallback


def create_fig2b_count_bootstrap(
    rec: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    i_sim: np.ndarray,
    v_sim: np.ndarray,
    labels: list[str],
    save_stem: str = "Fig2B",
) -> tuple[plt.Figure, plt.Axes]:
    """Create main hysteresis figure from the five-group mean after FFT filtering."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)

    i_sim_c, v_sim_c = close_loop(i_sim, v_sim)
    ax.plot(i_sim_c, v_sim_c, color="#F18F01", linewidth=3.0, label="Simulation", zorder=3)

    i_mean_c, v_mean_c = close_loop(rec["I_fft"], rec["V_fft"])
    # 先画浅色虚线
    ax.plot(
    i_mean_c,
    v_mean_c,
    "--",
    color="#1F4AFF",
    linewidth=1.5,
    alpha=0.25,
    zorder=2,
)

# 再画深色散点
    ax.scatter(
    i_mean_c,
    v_mean_c,
    s=38,
    color="#1F4AFF",
    edgecolors="black",
    linewidths=0.8,
    label="Experiment mean + FFT",
    zorder=3,
)
    area_nominal = loop_area(rec["I_fft"], rec["V_fft"])
    area_ci = np.percentile(boot["area"], [2.5, 97.5])
    area_text = (
    "$A_{\\mathrm{loop}}$ = %.2g\n"
    "$n$ = %d groups"
    % (area_nominal, len(labels))
)
    ax.text(
        0.98,
        0.05,
        area_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax.set_xlabel(r'$I_\mathrm{q}$', fontweight='bold', fontname='Arial', labelpad=5, color='black')
    ax.set_ylabel(r'$V_\mathrm{q}$', fontweight='bold', fontname='Arial', labelpad=5, color='black')

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    polish_axes(ax)
    style_legend(ax, fontsize=14, loc="best")
    apply_arial_to_figure(fig)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.96)
    save_figure(fig, OUTPUT_DIR / f"{save_stem}.svg")
    save_figure(fig, OUTPUT_DIR / f"{save_stem}.pdf")
    return fig, ax


def create_figs5_count_uncertainty(
    rec: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    t_sim: np.ndarray,
    v_sim: np.ndarray,
    save_stem: str = "FigS5",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create supplementary Vq uncertainty panel and exhaustive area histogram."""
    t_mid = rec["t_mid"]
    if boot["V"].shape[1] != t_mid.size:
        raise ValueError(
            f"Bootstrap Vq length {boot['V'].shape[1]} does not match t_mid length {t_mid.size}"
        )

    v_boot_95_low, v_boot_68_low, v_boot_68_high, v_boot_95_high = np.percentile(
        boot["V"], [2.5, 16.0, 84.0, 97.5], axis=0
    )

    fig, (ax_vq, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.8),
        dpi=200,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.35},
    )

    ax_vq.fill_between(
        t_mid,
        v_boot_95_low,
        v_boot_95_high,
        color="#1F4AFF",
        alpha=0.08,
        linewidth=0,
        label="95% exhaustive band",
        zorder=1,
    )
    ax_vq.fill_between(
        t_mid,
        v_boot_68_low,
        v_boot_68_high,
        color="#1F4AFF",
        alpha=0.18,
        linewidth=0,
        label="68% exhaustive band",
        zorder=2,
    )
    ax_vq.plot(t_mid, rec["V_raw"], color="0.25", alpha=0.48, linewidth=2.0, label="Raw mean")
    ax_vq.plot(t_sim[: len(v_sim)], v_sim, color="#F18F01", linewidth=2.6, label="Simulation")
    ax_vq.plot(t_mid, rec["V_fft"], color="#1F4AFF", linestyle="--", linewidth=2.8, label="FFT")
    ax_vq.set_xlabel(r'Time (μs)', fontname='Arial',
              
              fontstyle='normal',
              labelpad=5,
              color='black',fontsize=24)
    ax_vq.set_ylabel(r'$V_\mathrm{q}$', fontweight='bold', fontname='Arial', labelpad=5,fontsize=24, color='black')


    
    style_legend(ax_vq, fontsize=13, loc="best")
    polish_axes(ax_vq)
    ax_vq.yaxis.set_major_locator(MaxNLocator(6))
    ax_vq.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    area_nominal = loop_area(rec["I_fft"], rec["V_fft"])
    area_ci = np.percentile(boot["area"], [2.5, 97.5])
    ax_hist.hist(
        boot["area"],
        bins=34,
        color="#2E86AB",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.25,
    )
    ax_hist.axvline(area_nominal, color="black", linestyle="--", linewidth=1.4, label="Nominal area")
    ax_hist.axvspan(area_ci[0], area_ci[1], color="#1F4AFF", alpha=0.12, label="95% area interval")
    ax_hist.set_xlabel(r"$A_{\mathrm{loop}}$", fontweight="bold", fontname="Arial", labelpad=5, fontsize=24)
    ax_hist.set_ylabel("Exhaustive count",  fontname="Arial", labelpad=5, fontsize=24)
    style_legend(ax_hist, fontsize=13, loc="best")
    polish_axes(ax_hist)
    ax_hist.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))
    apply_arial_to_figure(fig)

    svg_path = save_figure(fig, OUTPUT_DIR / f"{save_stem}.svg")
    pdf_path = save_figure(fig, OUTPUT_DIR / f"{save_stem}.pdf")
    copy_figure(pdf_path, SUPPLEMENTARY_OUTPUT_DIR / "Fig.S5.pdf")
    copy_figure(pdf_path, SCIENCE_TEMPLATE_DIR / "Fig.S5.pdf")
    plt.show()
    return fig, (ax_vq, ax_hist)


def export_summary_tables(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    labels: list[str],
    rec: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    old_v_sem: np.ndarray,
    replicate_rec: dict[str, np.ndarray],
) -> None:
    """Export reconstruction diagnostics and plotted data."""
    diagnostics = validate_against_mean_csv(sx_reps, sy_reps)
    area_ci = np.percentile(boot["area"], [2.5, 16, 84, 97.5])
    summary = {
        **diagnostics,
        "n_groups": len(labels),
        "labels": ";".join(labels),
        "n_time_points_full": sx_reps.shape[1],
        "n_time_points_vq": rec["V_fft"].shape[0],
        "fft_cutoff_ratio_to_fmax": rec["cutoff_ratio"],
        "f_max_1_per_us": rec["f_max"],
        "fft_cutoff_1_per_us": rec["cutoff"],
        "loop_area_nominal": loop_area(rec["I_fft"], rec["V_fft"]),
        "loop_area_boot_mean": float(np.mean(boot["area"])),
        "loop_area_boot_std": float(np.std(boot["area"], ddof=1)),
        "loop_area_ci_2p5": float(area_ci[0]),
        "loop_area_ci_16": float(area_ci[1]),
        "loop_area_ci_84": float(area_ci[2]),
        "loop_area_ci_97p5": float(area_ci[3]),
        "median_abs_vq_raw_mean": float(np.median(np.abs(rec["V_raw"]))),
        "median_abs_vq_fft_mean": float(np.median(np.abs(rec["V_fft"]))),
        "median_old_pointwise_v_sem": float(np.median(old_v_sem)),
        "median_replicate_v_sem": float(np.median(replicate_rec["V_sem"])),
    }
    # pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "count_bootstrap_summary.csv", index=False)

    plotted = pd.DataFrame(
        {
            "t_us": rec["t_mid"],
            "Iq_raw_mean": rec["I_raw"],
            "Vq_raw_mean": rec["V_raw"],
            "Iq_fft_mean": rec["I_fft"],
            "Vq_fft_mean": rec["V_fft"],
            "Iq_group_sem_raw_replicates": replicate_rec["I_sem"],
            "Vq_group_sem_raw_replicates": replicate_rec["V_sem"],
            "Vq_old_pointwise_sem_reference": old_v_sem,
            "Vq_boot_2p5": np.percentile(boot["V"], 2.5, axis=0),
            "Vq_boot_16": np.percentile(boot["V"], 16, axis=0),
            "Vq_boot_84": np.percentile(boot["V"], 84, axis=0),
            "Vq_boot_97p5": np.percentile(boot["V"], 97.5, axis=0),
        }
    )
    # plotted.to_csv(OUTPUT_DIR / "count_bootstrap_reconstructed_loop.csv", index=False)

    replicate_rows = []
    for idx, label in enumerate(labels):
        for j, t_value in enumerate(replicate_rec["t_mid"]):
            replicate_rows.append(
                {
                    "group": label,
                    "t_us": t_value,
                    "Iq": replicate_rec["I_rep"][idx, j],
                    "Vq": replicate_rec["V_rep"][idx, j],
                }
            )
    # pd.DataFrame(replicate_rows).to_csv(OUTPUT_DIR / "count_replicate_iq_vq.csv", index=False)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with a guard against zero-variance vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Correlation arrays must have the same shape. Got {x.shape} and {y.shape}.")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def export_cutoff_robustness(
    sx_reps: np.ndarray,
    sy_reps: np.ndarray,
    ratios: tuple[float, ...] = ROBUSTNESS_RATIOS,
    nominal_ratio: float = ROBUSTNESS_NOMINAL_RATIO,
) -> pd.DataFrame:
    
    nominal_rec = reconstruct_mean_then_fft_loop(sx_reps, sy_reps, cutoff_ratio=nominal_ratio)
    nominal_area_raw = loop_area(nominal_rec["I_fft"], nominal_rec["V_fft"])

    # Use the nominal loop orientation as the sign convention. If the numerical
    # integration returns a negative nominal signed area because of traversal
    # direction, flip all areas consistently so the table reports positive
    # signed areas while preserving relative changes.
    orientation = -1.0 if nominal_area_raw >= 0 else 1.0
    nominal_area = orientation * nominal_area_raw
    if np.isclose(nominal_area, 0.0):
        raise ValueError("Nominal loop area is too close to zero for relative-area normalization.")

    rows = []
    for ratio in ratios:
        rec = reconstruct_mean_then_fft_loop(sx_reps, sy_reps, cutoff_ratio=ratio)
        signed_area = orientation * loop_area(rec["I_fft"], rec["V_fft"])
        relative_area_change = 100.0 * (signed_area - nominal_area) / abs(nominal_area)
        corr_vq = safe_corrcoef(rec["V_fft"], nominal_rec["V_fft"])
        corr_iq = safe_corrcoef(rec["I_fft"], nominal_rec["I_fft"])
        rows.append(
            {
                "cutoff": f"{ratio:.2f} f_max",
                "cutoff_ratio_to_fmax": ratio,
                "signed_area": signed_area,
                "relative_area_change_percent": relative_area_change,
                "corr_with_nominal_Vq": corr_vq,
                "corr_with_nominal_Iq": corr_iq,
            }
        )

    df = pd.DataFrame(rows)

    # Numeric CSV for analysis / table generation. Column order exactly follows
    # the LaTeX table content.
    numeric_cols = [
        "cutoff",
        "signed_area",
        "relative_area_change_percent",
        "corr_with_nominal_Vq",
        "corr_with_nominal_Iq",
    ]
    df[numeric_cols].to_csv(OUTPUT_DIR / "cutoff_robustness.csv", index=False)

    
 
    return df[numeric_cols]


def main() -> None:
    set_science_style()
    sx_reps, sy_reps, labels = load_stokes_replicates()
    validation = validate_against_mean_csv(sx_reps, sy_reps)
    print("Loaded Stokes CSV groups:", labels)
    print("Reconstruction validation:", validation)

    replicate_rec = reconstruct_iq_vq_replicates(sx_reps, sy_reps)
    rec = reconstruct_mean_then_fft_loop(sx_reps, sy_reps)
    if rec["V_fft"].shape[0] != sx_reps.shape[1] - 2:
        raise RuntimeError("Vq length mismatch after center-difference reconstruction.")

    boot = group_bootstrap_mean_then_fft(sx_reps, sy_reps)
    if not np.isfinite(boot["I"]).all() or not np.isfinite(boot["V"]).all() or not np.isfinite(boot["area"]).all():
        raise RuntimeError("Bootstrap output contains NaN or inf.")

    old_v_sem = old_pointwise_vq_sem(sx_reps, sy_reps)
    t_sim, i_sim, v_sim = run_qutip_simulation()
    if len(t_sim) != len(i_sim) or len(i_sim) != len(v_sim):
        raise RuntimeError("Simulation output arrays have mismatched lengths.")

    create_fig2b_count_bootstrap(rec, boot, i_sim, v_sim, labels)
    create_figs5_count_uncertainty(rec, boot, t_sim, v_sim)
    export_summary_tables(sx_reps, sy_reps, labels, rec, boot, old_v_sem, replicate_rec)
    robustness_df = export_cutoff_robustness(sx_reps, sy_reps)

    print("Saved exhaustive-bootstrap figures and tables to:", OUTPUT_DIR)
    print("Vq length:", rec["V_fft"].shape[0])
    print("FFT cutoff ratio to f_max:", rec["cutoff_ratio"])
    print("FFT f_max:", rec["f_max"])
    print("Loop area nominal:", loop_area(rec["I_fft"], rec["V_fft"]))
    print("Loop area 95% CI:", np.percentile(boot["area"], [2.5, 97.5]))
    print("FFT cutoff robustness:")
    print(robustness_df)


if __name__ == "__main__":
    main()
