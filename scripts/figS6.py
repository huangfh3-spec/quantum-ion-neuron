
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.optimize import curve_fit

# ===================== paths =====================
try:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from paths import DATA  # type: ignore
except Exception:
    DATA = Path(__file__).resolve().parent

HERE = Path('/mnt/data')  # always writable

# ===================== style =====================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.unicode_minus': False,
    'axes.linewidth': 1.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ===================== IO helpers =====================
def read_waveform(path: Path):
    df = pd.read_csv(path)
    return (
        df.iloc[:, 0].to_numpy(dtype=float),
        df.iloc[:, 1].to_numpy(dtype=float),
        df.iloc[:, 2].to_numpy(dtype=float),
    )

def read_three_col_numeric(path: Path):
    try:
        df = pd.read_csv(path)
        df = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) == 0:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna()
    return (
        df.iloc[:, 0].to_numpy(dtype=float),
        df.iloc[:, 1].to_numpy(dtype=float),
        df.iloc[:, 2].to_numpy(dtype=float),
    )

def exp_decay_model(t, gamma):
    return 0.25 * np.exp(-gamma * t)

def fit_gamma(pop_path: Path):
    t, p, _ = read_three_col_numeric(pop_path)
    popt, pcov = curve_fit(exp_decay_model, t, p, p0=[0.005], maxfev=100000)
    gamma = float(popt[0])
    gamma_err = float(math.sqrt(pcov[0, 0])) if pcov.size else float('nan')
    return gamma, gamma_err

def fft_lowpass(signal, cutoff, dt):
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=dt)
    mask = np.abs(freq) < cutoff
    return np.fft.ifft(fft * mask).real

def polygon_area(x, y):
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

# ===================== FFT robustness =====================
def analyze_fft_robustness():
    sx = pd.read_csv(DATA / "sx_mean.csv", header=None).apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    sy = pd.read_csv(DATA / "sy_mean.csv", header=None).apply(pd.to_numeric, errors="coerce").dropna().to_numpy()

    sx_mean = sx[:, 0]
    sx_err = sx[:, 1]
    sy_mean = sy[:, 0]
    sy_err = sy[:, 1]

    step = 4.0
    n = len(sx_mean)

    sx_used = sx_mean[1:n-1]
    sy_used = sy_mean[1:n-1]

    dy_dt = (sy_mean[2:] - sy_mean[:-2]) / (2 * step)

    v_raw = dy_dt - 0.04 * sx_used
    i_raw = -0.75 * sy_used

    freq = np.fft.fftfreq(len(v_raw), d=step)
    fmax = float(np.max(freq))
    nominal_fraction = 0.35

    v_nom = fft_lowpass(v_raw, nominal_fraction * fmax, step)
    i_nom = fft_lowpass(i_raw, nominal_fraction * fmax, step)
    area_nom = polygon_area(i_nom, v_nom)

    rows = []
    for frac in [0.25, 0.30, 0.35, 0.40, 0.45]:
        cutoff = frac * fmax
        v_f = fft_lowpass(v_raw, cutoff, step)
        i_f = fft_lowpass(i_raw, cutoff, step)
        area = polygon_area(i_f, v_f)
        rows.append(
            {
                "Cutoff": f"{frac:.2f}$f_{{\\max}}$",
                "Signed area": area,
                "Relative area change (%)": (area / area_nom - 1.0) * 100 if area_nom != 0 else np.nan,
                "Corr. with nominal Vq": float(np.corrcoef(v_nom, v_f)[0, 1]),
                "Corr. with nominal Iq": float(np.corrcoef(i_nom, i_f)[0, 1]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fft_cutoff_robustness.csv", index=False)
    return df, {"fmax": fmax, "nominal_fraction": nominal_fraction}

# ===================== XOR robustness =====================
def build_cases():
    t_high, i_high, _ = read_waveform(DATA / 'I_high_waveform.csv')
    t_low, i_low, _ = read_waveform(DATA / 'I_low_waveform.csv')

    cases = {
        'C1': {'wave_t': t_high, 'wave_i': i_high, 'pop': DATA / 'High_Low_data.csv', 'expected': 1},
        'C2': {'wave_t': t_high, 'wave_i': i_high, 'pop': DATA / 'High_High_data.csv', 'expected': 0},
        'C3': {'wave_t': t_low,  'wave_i': i_low,  'pop': DATA / 'Low_Low_data.csv',  'expected': 0},
        'C4': {'wave_t': t_low,  'wave_i': i_low,  'pop': DATA / 'Low_High_data.csv', 'expected': 1},
    }
    for _, info in cases.items():
        gamma, gamma_err = fit_gamma(info['pop'])
        info['gamma'] = gamma
        info['gamma_err'] = gamma_err
        info['v_exp'] = gamma * info['wave_i']
    return cases

CASES = build_cases()
BASELINE_B = 1.6e-3
BASELINE_WINDOW = (45, 65)

COLORS = {
    'C1': '#F0850A',
    'C2': '#D93A49',
    'C3': '#50647F',
    'C4': '#009688',
}
LABELS = {
    'C1': 'C1 (firing)',
    'C2': 'C2 (no firing)',
    'C3': 'C3 (no firing)',
    'C4': 'C4 (firing)',
}

def classify(case_key: str, start: float, end: float, b: float):
    info = CASES[case_key]
    mask = (info['wave_t'] >= start) & (info['wave_t'] <= end)
    margin = float(np.max(info['v_exp'][mask]) - b)
    pred = int(margin >= 0)
    return pred, margin

def signed_margin(case_key: str, start: float, end: float, b: float):
    _, margin = classify(case_key, start, end, b)
    expected = CASES[case_key]['expected']
    return margin if expected == 1 else -margin

def analyze_xor_robustness():
    threshold_scan = []
    for b in np.linspace(0.0010, 0.0022, 301):
        preds = {k: classify(k, BASELINE_WINDOW[0], BASELINE_WINDOW[1], b)[0] for k in CASES}
        ok = all(preds[k] == CASES[k]["expected"] for k in CASES)
        threshold_scan.append({"threshold": b, "stable": ok})

    window_scan = []
    for start in range(40, 50):
        for end in range(60, 70):
            preds = {k: classify(k, start, end, BASELINE_B)[0] for k in CASES}
            ok = all(preds[k] == CASES[k]["expected"] for k in CASES)
            window_scan.append({"start_us": start, "end_us": end, "stable": ok})

    margin_rows = []
    for key in ['C1', 'C2', 'C3', 'C4']:
        pred, margin = classify(key, BASELINE_WINDOW[0], BASELINE_WINDOW[1], BASELINE_B)
        margin_rows.append(
            {
                "case": key,
                "gamma_fit": CASES[key]["gamma"],
                "gamma_err": CASES[key]["gamma_err"],
                "predicted_output": pred,
                "expected_output": CASES[key]["expected"],
                "raw_margin": margin,
            }
        )

    threshold_df = pd.DataFrame(threshold_scan)
    window_df = pd.DataFrame(window_scan)
    margins_df = pd.DataFrame(margin_rows)

    threshold_df.to_csv(DATA / "xor_threshold_scan.csv", index=False)
    window_df.to_csv(DATA / "xor_window_scan.csv", index=False)
    margins_df.to_csv(DATA / "xor_margin_summary.csv", index=False)
    return margins_df, threshold_df, window_df

def plot_xor_combined_pdf():
    margins_df, threshold_df, window_df = analyze_xor_robustness()

    fig = plt.figure(figsize=(13.2, 5.6), dpi=200)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    thresholds = threshold_df["threshold"].to_numpy()
    raw_margin_map = {k: [] for k in CASES}
    all_stable = []

    for b in thresholds:
        ok = True
        for key in CASES:
            pred, margin = classify(key, BASELINE_WINDOW[0], BASELINE_WINDOW[1], b)
            raw_margin_map[key].append(margin)
            if pred != CASES[key]["expected"]:
                ok = False
        all_stable.append(ok)

    stable_thresholds = thresholds[np.array(all_stable, dtype=bool)]
    if len(stable_thresholds) > 0:
        ax1.axvspan(stable_thresholds.min() * 1e3, stable_thresholds.max() * 1e3,
                    alpha=0.20, color='#B0BEC5', zorder=0)

    for key in ['C1', 'C2', 'C3', 'C4']:
        ax1.plot(
            thresholds * 1e3,
            np.array(raw_margin_map[key]) * 1e4,
            color=COLORS[key],
            linewidth=2.5,
            alpha=0.98,
            solid_capstyle='round',
            zorder=3,
            label=LABELS[key]
        )

    ax1.axhline(y=0, color='#A61B29', linestyle='--', linewidth=1.8, zorder=2)
    ax1.axvline(x=BASELINE_B * 1e3, color='#6A1B9A', linestyle=':', linewidth=1.8, zorder=2)

    for key in ['C1', 'C2', 'C3', 'C4']:
        _, base_margin = classify(key, BASELINE_WINDOW[0], BASELINE_WINDOW[1], BASELINE_B)
        ax1.plot(
            [BASELINE_B * 1e3], [base_margin * 1e4],
            marker='o', markersize=5.6,
            color=COLORS[key], markeredgecolor='black', markeredgewidth=0.8, zorder=5
        )

    ylim = ax1.get_ylim()
    if len(stable_thresholds) > 0:
        midx = 0.5 * (stable_thresholds.min() + stable_thresholds.max()) * 1e3
        ax1.text(
            midx, ylim[1] - 0.08 * (ylim[1] - ylim[0]),
            f'Stable XOR region: {stable_thresholds.min()*1e3:.2f}–{stable_thresholds.max()*1e3:.2f}',
            ha='center', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.92),
        )
    ax1.text(
        0.985, 0.965, 'Above 0: firing\nBelow 0: no firing',
        transform=ax1.transAxes, ha='right', va='top', fontsize=8.6,
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='gray', alpha=0.92),
    )

    ax1.set_xlabel(r'Threshold b ($\times 10^{-3}$)',fontweight='bold')
    ax1.set_ylabel(r'Raw margin $\kappa$ ($\times 10^{-4}$)',fontweight='bold')
    ax1.xaxis.set_major_locator(MaxNLocator(7))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, alpha=0.24, linestyle='-', linewidth=0.6)
    ax1.grid(True, which='minor', alpha=0.14, linestyle='--', linewidth=0.4)
    ax1.set_facecolor('white')
    ax1.tick_params(axis='both', which='major', length=4.5, width=1.4)
    ax1.tick_params(axis='both', which='minor', length=2.5, width=0.8)

    legend_handles = [
        Line2D([0], [0], color=COLORS['C1'], lw=2.5, label=LABELS['C1']),
        Line2D([0], [0], color=COLORS['C2'], lw=2.5, label=LABELS['C2']),
        Line2D([0], [0], color=COLORS['C3'], lw=2.5, label=LABELS['C3']),
        Line2D([0], [0], color=COLORS['C4'], lw=2.5, label=LABELS['C4']),
        Line2D([0], [0], color='#A61B29', lw=1.8, linestyle='--', label='Decision boundary'),
        Line2D([0], [0], color='#6A1B9A', lw=1.8, linestyle=':', label='Baseline threshold'),
    ]
    ax1.legend(
        handles=legend_handles,
        loc='lower left',
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor='gray',
        facecolor='white',
        borderpad=0.28,
        labelspacing=0.24,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=0.75,
        fontsize=8.8,
    )

    starts = np.arange(40, 50)
    ends = np.arange(60, 70)
    robust_grid = np.zeros((len(ends), len(starts)))
    for i, end in enumerate(ends):
        for j, start in enumerate(starts):
            sms = [signed_margin(key, start, end, BASELINE_B) for key in CASES]
            robust_grid[i, j] = min(sms) * 1e4

    vmin = float(np.min(robust_grid))
    vmax = float(np.max(robust_grid))
    if vmin == vmax:
        vmin -= 1
        vmax += 1

    im = ax2.imshow(
        robust_grid,
        origin='lower',
        aspect='auto',
        extent=[starts.min()-0.5, starts.max()+0.5, ends.min()-0.5, ends.max()+0.5],
        cmap='viridis',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    ax2.plot(
        BASELINE_WINDOW[0], BASELINE_WINDOW[1],
        marker='s', markersize=6.5,
        markerfacecolor='none', markeredgecolor='#C62828',
        markeredgewidth=1.8, zorder=3
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'Minimum signed margin $\mathbf{\kappa}$ ($\mathbf{\times 10^{-4}}$)', fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=11, width=1.2, length=3.5)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.2)

    ax2.set_xlabel(r'Window start time ($\mathrm{\mu}$s)',fontweight='bold')
    ax2.set_ylabel(r'Window end time ($\mathrm{\mu}$s)',fontweight='bold')
    ax2.set_xticks(starts)
    ax2.set_yticks(ends)
    ax2.set_facecolor('white')
    ax2.grid(True, alpha=0.48, linestyle='--', linewidth=0.45)
    ax2.tick_params(axis='both', which='major', length=4.5, width=1.4)

    for spine in ax2.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color('gray')

    ax2.legend(
        handles=[
            Line2D([0], [0], marker='s', color='none', markeredgecolor='#C62828',
                   markerfacecolor='none', markeredgewidth=1.8, markersize=6.3,
                   label='Baseline window')
        ],
        loc='upper left',
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor='gray',
        facecolor='white',
        borderpad=0.28,
        labelspacing=0.22,
        handlelength=1.0,
        handletextpad=0.45,
        fontsize=8.8,
    )

    ax1.text(-0.11, 1.03, 'A', transform=ax1.transAxes, fontsize=13, fontweight='bold')
    ax2.text(-0.11, 1.03, 'B', transform=ax2.transAxes, fontsize=13, fontweight='bold')

    fig.patch.set_facecolor('white')
    fig.tight_layout()

    out_pdf = DATA / 'Fig.S6.pdf'
    fig.savefig(out_pdf, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close(fig)
    return out_pdf, margins_df, threshold_df, window_df

def main():
    fft_df, fft_meta = analyze_fft_robustness()
    out_pdf, margins_df, threshold_df, window_df = plot_xor_combined_pdf()

    print("=== FFT robustness ===")
    printable = fft_df.copy()
    printable["Signed area"] = printable["Signed area"].map(lambda x: f"{x:.4e}")
    printable["Relative area change (%)"] = printable["Relative area change (%)"].map(lambda x: f"{x:+.2f}%")
    printable["Corr. with nominal Vq"] = printable["Corr. with nominal Vq"].map(lambda x: f"{x:.4f}")
    printable["Corr. with nominal Iq"] = printable["Corr. with nominal Iq"].map(lambda x: f"{x:.4f}")
    print(printable.to_string(index=False))
    print(f"Nominal cutoff fraction: {fft_meta['nominal_fraction']:.2f} fmax")
    print()

    stable_thresholds = threshold_df.loc[threshold_df["stable"], "threshold"]
    stable_windows = window_df.loc[window_df["stable"]]
    print("=== XOR local robustness summary ===")
    if len(stable_thresholds) > 0:
        print(f"Stable threshold range: {stable_thresholds.min():.5f} to {stable_thresholds.max():.5f}")
    if len(stable_windows) > 0:
        print(
            "Stable window range: start "
            f"{int(stable_windows['start_us'].min())} to {int(stable_windows['start_us'].max())} us, end "
            f"{int(stable_windows['end_us'].min())} to {int(stable_windows['end_us'].max())} us"
        )
    print(f"Saved combined XOR figure: {out_pdf}")
    
if __name__ == '__main__':
    main()
