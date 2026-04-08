"""
Plot error distributions between reference data and torchkrates-eval predictions.

Loads one or more eval HDF5 files produced by `torchkrates-eval`, computes
per-component error statistics (MAE, RMSE, mean bias, MAD), and plots the
error distributions as histograms. For vector properties (forces, stress,
dipole) each Cartesian component is shown in a separate panel.

Multiple eval files can be overlaid in a single plot (--layout overlay, default)
or displayed as one row of panels per model (--layout subplots).

Usage examples
--------------
# Single eval file, force errors:
    python plot_errors.py --data ref.xyz --eval preds.h5 --property forces

# Compare two models, overlay:
    python plot_errors.py --data ref.xyz \\
        --eval model1.h5 --label "Model v1" \\
        --eval model2.h5 --label "Model v2" \\
        --property forces --layout overlay --output compare.png

# Side-by-side subplots + text log:
    python plot_errors.py --data ref.xyz \\
        --eval m1.h5 --eval m2.h5 \\
        --property energy_per_atom --layout subplots \\
        --output compare.png --log metrics.txt
"""

import argparse
import logging
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Wong (2011) 8-colour palette — safe for protanopia, deuteranopia,
# tritanopia. Order chosen for maximum perceptual distance.
COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]

# ── property metadata ────────────────────────────────────────────────────────

PROPERTY_META = {
    "energy_per_atom": {
        "ref_type": "info",  # atoms.info or atoms.arrays
        "ref_key_arg": "energy_key",  # which CLI arg holds the key name
        "pred_hdf5_key": "energies",
        "components": ["E/atom"],
        "unit": "meV/atom",
        "scale": 1000.0,  # eV → meV
        "per_atom": True,
    },
    "forces": {
        "ref_type": "arrays",
        "ref_key_arg": "forces_key",
        "pred_hdf5_key": "forces",
        "components": ["F"],
        "unit": "meV/Å",
        "scale": 1000.0,
        "per_atom": False,
    },
    "stress": {
        "ref_type": "info",
        "ref_key_arg": "stress_key",
        "pred_hdf5_key": "stresses",
        "components": ["xx", "yy", "zz", "xy", "xz", "yz"],
        "unit": "eV/Å³",
        "scale": 1.0,
        "per_atom": False,
    },
    "dipole": {
        "ref_type": "info",
        "ref_key_arg": "dipole_key",
        "pred_hdf5_key": "dipoles",
        "components": ["x", "y", "z"],
        "unit": "eÅ",
        "scale": 1.0,
        "per_atom": False,
    },
    "hirshfeld_ratios": {
        "ref_type": "arrays",
        "ref_key_arg": "hirshfeld_key",
        "pred_hdf5_key": "hirshfeld_ratios",
        "components": ["ratio"],
        "unit": "",
        "scale": 1.0,
        "per_atom": False,
    },
    "charges": {
        "ref_type": "arrays",
        "ref_key_arg": "charges_key",
        "pred_hdf5_key": "partial_charges",
        "components": ["q"],
        "unit": "e",
        "scale": 1.0,
        "per_atom": False,
    },
}


# ── reference loading ────────────────────────────────────────────────────────


def load_reference(data_path, prop, keys):
    """Load reference values from a data file.

    Returns (ref_array, n_atoms_list).
    ref_array shape: [N] for scalars, [N_total, n_components] for vectors.
    """
    logging.info(f"Loading reference data from {data_path}")

    if data_path.endswith((".h5", ".hdf5")):
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

        atoms_list = load_atoms_from_hdf5(data_path, index=None)
    else:
        import ase.io

        atoms_list = ase.io.read(data_path, index=":")

    logging.info(f"Loaded {len(atoms_list)} reference structures")

    meta = PROPERTY_META[prop]
    ref_key = keys[meta["ref_key_arg"]]

    # Since ASE >= 3.23.0b1, bare keys like "forces", "energy", "stress"
    # are absorbed into atoms.calc rather than atoms.info / atoms.arrays.
    # Re-materialise them the same way the training pipeline does.
    _ASE_CALC_GETTERS = {
        "energy": lambda a: a.get_potential_energy(),
        "forces": lambda a: a.get_forces(),
        "stress": lambda a: a.get_stress(),
    }
    if ref_key in _ASE_CALC_GETTERS:
        logging.info(
            f"Key '{ref_key}' may be absorbed by the ASE calculator; "
            f"extracting via atoms.get_{ref_key}()."
        )
        remap_key = f"_plot_ref_{ref_key}"
        getter = _ASE_CALC_GETTERS[ref_key]
        for atoms in atoms_list:
            try:
                val = getter(atoms)
            except Exception:
                val = None
            if val is not None:
                if np.ndim(val) == 0 or len(np.shape(val)) <= 1:
                    atoms.info[remap_key] = val
                else:
                    atoms.arrays[remap_key] = np.asarray(val)
        ref_key = remap_key

    n_atoms_list = [len(a) for a in atoms_list]
    values = []

    for atoms in atoms_list:
        if meta["ref_type"] == "info":
            val = atoms.info.get(ref_key, None)
            if val is None:
                val = atoms.arrays.get(ref_key, None)
        else:
            val = atoms.arrays.get(ref_key, None)
            if val is None:
                val = atoms.info.get(ref_key, None)

        if val is None:
            available_info = list(atoms.info.keys())
            available_arrays = list(atoms.arrays.keys())
            raise ValueError(
                f"Reference property '{ref_key}' not found in structure. "
                f"Available info keys: {available_info}. "
                f"Available array keys: {available_arrays}. "
                f"Check --{meta['ref_key_arg']} to specify the correct key."
            )
        values.append(np.asarray(val, dtype=np.float64))

    if meta["per_atom"]:
        # scalar per structure → divide by N_atoms
        ref = np.array(
            [v.item() / n for v, n in zip(values, n_atoms_list)],
            dtype=np.float64,
        )
    elif prop == "stress":
        # stress may be [3,3] or [1,3,3] — extract upper triangle
        rows = []
        for v in values:
            v = np.asarray(v).reshape(3, 3)
            rows.append([v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2]])
        ref = np.array(rows, dtype=np.float64)  # [N_structures, 6]
    else:
        ref = np.vstack(values)  # [N_total_atoms, n_components] or [N, 3]

    return ref, n_atoms_list


# ── prediction loading ───────────────────────────────────────────────────────


def load_predictions(eval_path, prop, n_atoms_list):
    """Load predictions from a torchkrates-eval HDF5 output file.

    Handles both single-model and ensemble outputs. For ensembles the
    per-model arrays are averaged before returning.
    """
    meta = PROPERTY_META[prop]
    hdf5_key = meta["pred_hdf5_key"]

    with h5py.File(eval_path, "r") as f:
        if hdf5_key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Key '{hdf5_key}' not found in {eval_path}. "
                f"Available keys: {available}. "
                f"Did you pass the right --property?"
            )

        item = f[hdf5_key]
        if isinstance(item, h5py.Group):
            arrays = []
            for key in sorted(item.keys()):
                grp = item[key]
                if isinstance(grp, h5py.Dataset):
                    arrays.append(grp[()])
                elif "result" in grp.keys():
                    arrays.append(grp["result"][()])
                else:
                    # Stored as item_000000, item_000001, …
                    parts = [grp[k][()] for k in sorted(grp.keys())]
                    arrays.append(np.concatenate(parts, axis=0))
            # If all arrays share the same shape it's a multi-head ensemble
            # → stack and average. Otherwise arrays are per-molecule (ragged
            # per-atom data for molecules with different sizes) → concatenate.
            shapes = {a.shape for a in arrays}
            if len(shapes) == 1:
                pred = np.mean(np.stack(arrays, axis=0), axis=0)
                logging.info(
                    f"Loaded ensemble ({len(arrays)} models) "
                    f"from {eval_path}"
                )
            else:
                pred = np.concatenate(arrays, axis=0)
                logging.info(
                    f"Loaded single-model predictions from {eval_path}"
                )
        else:
            pred = item[()]
            logging.info(f"Loaded single-model predictions from {eval_path}")

    pred = pred.astype(np.float64)

    # Forces/hirshfeld/charges may be stored as (N_structures, N_atoms, C)
    # when all structures have the same atom count (single-species dataset).
    # Flatten to (N_total_atoms, C) to match the vstack'd reference shape.
    if pred.ndim == 3 and not meta["per_atom"] and prop != "stress":
        pred = pred.reshape(-1, pred.shape[-1])

    if meta["per_atom"]:
        n_atoms = np.array(n_atoms_list, dtype=np.float64)
        pred = pred / n_atoms  # [N_structures]
    elif prop == "stress":
        # pred shape [N, 3, 3] or [N, 1, 3, 3] → extract upper triangle
        pred = pred.reshape(-1, 3, 3)
        pred = np.stack(
            [
                pred[:, 0, 0],
                pred[:, 1, 1],
                pred[:, 2, 2],
                pred[:, 0, 1],
                pred[:, 0, 2],
                pred[:, 1, 2],
            ],
            axis=-1,
        )  # [N_structures, 6]

    return pred


# ── statistics ───────────────────────────────────────────────────────────────


def compute_stats(errors):
    """Compute scalar error statistics over a flat array of errors."""
    from so3krates_torch.tools.utils import (
        compute_mae,
        compute_rmse,
        compute_q95,
    )

    flat = errors.flatten()
    mean = float(np.mean(flat))
    mad = float(np.mean(np.abs(flat - mean)))
    return {
        "mae": compute_mae(flat),
        "rmse": compute_rmse(flat),
        "mean": mean,
        "mad": mad,
        "std": float(np.std(flat)),
        "q95": compute_q95(flat),
        "n": len(flat),
    }


def compute_per_component_stats(errors, components):
    """Return list of stats dicts, one per component, plus one for 'all'."""
    n_comp = errors.shape[1] if errors.ndim > 1 else 1
    result = []
    for i in range(n_comp):
        col = errors[:, i] if errors.ndim > 1 else errors
        result.append(compute_stats(col))
    result.append(compute_stats(errors))  # "all" components together
    return result  # length n_comp + 1


# ── plotting ─────────────────────────────────────────────────────────────────


def _stats_text(stats, unit):
    unit_str = f" {unit}" if unit else ""
    return (
        f"MAE  = {stats['mae']:.4g}{unit_str}\n"
        f"RMSE = {stats['rmse']:.4g}{unit_str}\n"
        f"Mean = {stats['mean']:.4g}{unit_str}\n"
        f"MAD  = {stats['mad']:.4g}{unit_str}"
    )


def _annotate_stats(ax, stats, unit, color, y_frac=0.97, x_frac=0.97):
    """Place a stats text box in the upper-right corner of ax."""
    ax.text(
        x_frac,
        y_frac,
        _stats_text(stats, unit),
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            alpha=0.85,
        ),
        color=color,
        family="monospace",
    )


def _plot_ccdf_row(
    axes_row, errors_list_scaled, labels, components, unit, colors
):
    """Plot CCDF of |error| per component column on log-logit axes."""
    _PERCENTILES = [
        (0.50, "50th"),
        (0.90, "90th"),
        (0.99, "99th"),
        (0.999, "99.9th"),
    ]
    for c_idx, comp in enumerate(components):
        ax = axes_row[c_idx]
        min_abs = None
        max_abs = None
        for m_idx, errors in enumerate(errors_list_scaled):
            col = (errors[:, c_idx] if errors.ndim > 1 else errors).flatten()
            abs_errs = np.abs(col)
            abs_errs = abs_errs[abs_errs > 0]
            if len(abs_errs) == 0:
                continue
            x = np.sort(abs_errs)
            n = len(x)
            y = (n - np.arange(n, dtype=float)) / n
            ax.step(
                x,
                y,
                where="post",
                linewidth=1.8,
                color=colors[m_idx],
                label=labels[m_idx],
            )
            this_min = float(x[0])
            this_max = float(x[-1])
            min_abs = this_min if min_abs is None else min(min_abs, this_min)
            max_abs = this_max if max_abs is None else max(max_abs, this_max)
        if min_abs is not None and max_abs is not None:
            ax.set_xscale("log")
            ax.set_yscale("logit")
            ax.set_xlim(min_abs, max_abs)
            ax.grid(
                True,
                which="major",
                axis="x",
                color="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3,
            )
            for pct_val, pct_label in _PERCENTILES:
                y_val = 1.0 - pct_val
                ax.axhline(
                    y_val,
                    color="black",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.5,
                )
                ax.text(
                    max_abs * 0.98,
                    y_val,
                    pct_label,
                    fontsize=6,
                    va="bottom",
                    ha="right",
                    color="black",
                )
        ax.set_xlabel(f"|Δ{comp}| ({unit})", fontsize=8)
        ax.set_ylabel("P(|Δ| > x)", fontsize=8)
        if c_idx == 0:
            ax.legend(fontsize=7, frameon=True, framealpha=0.85)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))


def build_plot(errors_list, stats_list, labels, prop, layout, bins):
    """Build and return a matplotlib Figure.

    errors_list : list of np.ndarray, one per eval file
    stats_list  : list of per-component stats (from compute_per_component_stats)
    labels      : list of str, one per eval file
    prop        : property name (key in PROPERTY_META)
    layout      : "overlay" or "subplots"
    bins        : int
    """
    meta = PROPERTY_META[prop]
    components = meta["components"]
    unit = meta["unit"]
    scale = meta["scale"]
    n_comp = len(components)
    n_models = len(errors_list)

    # Apply unit scale
    errors_list_scaled = [e * scale for e in errors_list]
    model_colors = [COLORS[i % len(COLORS)] for i in range(n_models)]

    if layout == "overlay":
        fig, all_axes = plt.subplots(
            2,
            n_comp,
            figsize=(3.5 * n_comp, 7.0),
            squeeze=False,
        )
        axes = all_axes[0]  # histogram row

        for m_idx, (errors, stats_per_comp) in enumerate(
            zip(errors_list_scaled, stats_list)
        ):
            color = model_colors[m_idx]
            label = labels[m_idx]

            for c_idx, (comp, ax) in enumerate(zip(components, axes)):
                col = errors[:, c_idx] if errors.ndim > 1 else errors
                comp_st = {
                    k: v * scale if k not in ("n",) else v
                    for k, v in stats_per_comp[c_idx].items()
                }
                unit_str = f" {unit}" if unit else ""
                rich_label = (
                    f"{label}\n"
                    f"MAE  = {comp_st['mae']:.4g}{unit_str}\n"
                    f"RMSE = {comp_st['rmse']:.4g}{unit_str}\n"
                    f"Mean = {comp_st['mean']:.4g}{unit_str}\n"
                    f"MAD  = {comp_st['mad']:.4g}{unit_str}"
                )
                ax.hist(
                    col,
                    bins=bins,
                    alpha=0.55,
                    color=color,
                    label=rich_label,
                    density=True,
                    linewidth=0,
                )
                ax.set_xlabel(f"Δ{comp} ({unit})")
                ax.set_ylabel("Density")
                ax.set_title(comp)
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

        for ax in axes:
            ax.legend(
                fontsize=7,
                loc="upper left",
                framealpha=0.85,
                handlelength=1.0,
                borderpad=0.5,
            )

        _plot_ccdf_row(
            all_axes[1],
            errors_list_scaled,
            labels,
            components,
            unit,
            model_colors,
        )

    else:  # subplots: one row per model, plus CCDF overlay row
        fig, axes = plt.subplots(
            n_models + 1,
            n_comp,
            figsize=(3.5 * n_comp, 2.8 * (n_models + 1)),
            squeeze=False,
        )
        # Compute shared x-range per component for histogram rows
        x_min = np.full(n_comp, np.inf)
        x_max = np.full(n_comp, -np.inf)
        for errors in errors_list_scaled:
            for c_idx in range(n_comp):
                col = errors[:, c_idx] if errors.ndim > 1 else errors
                x_min[c_idx] = min(x_min[c_idx], col.min())
                x_max[c_idx] = max(x_max[c_idx], col.max())

        for m_idx, (errors, stats_per_comp) in enumerate(
            zip(errors_list_scaled, stats_list)
        ):
            color = model_colors[m_idx]
            label = labels[m_idx]

            for c_idx, comp in enumerate(components):
                ax = axes[m_idx][c_idx]
                col = errors[:, c_idx] if errors.ndim > 1 else errors
                ax.hist(
                    col,
                    bins=bins,
                    color=color,
                    density=True,
                    linewidth=0,
                )
                _annotate_stats(
                    ax,
                    {
                        k: v * scale if k not in ("n",) else v
                        for k, v in stats_per_comp[c_idx].items()
                    },
                    unit,
                    color,
                )
                if m_idx == 0:
                    ax.set_title(comp)
                if m_idx == n_models - 1:
                    ax.set_xlabel(f"Δ{comp} ({unit})")
                ax.set_ylabel(f"{label}\nDensity", fontsize=7)
                ax.set_xlim(x_min[c_idx], x_max[c_idx])
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

        _plot_ccdf_row(
            axes[n_models],
            errors_list_scaled,
            labels,
            components,
            unit,
            model_colors,
        )
        axes[n_models, 0].set_ylabel(
            "CCDF — all models\nP(|Δ| > x)",
            fontsize=7,
        )

    fig.suptitle(
        f"Error distribution — {prop.replace('_', ' ')}",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ── metrics log ──────────────────────────────────────────────────────────────


def write_metrics_table(stats_list, labels, components, unit, scale, log_path):
    """Write a human-readable metrics table to a text file."""
    col_names = components + ["all"]
    header = (
        f"{'Model':<20}  {'Component':<12}  "
        f"{'MAE':>10}  {'RMSE':>10}  "
        f"{'Mean':>10}  {'MAD':>10}  {'Q95':>10}  {'N':>8}"
    )
    sep = "-" * len(header)

    lines = [
        f"Property : {components[0] if len(components) == 1 else 'see components below'}",
        f"Unit     : {unit}",
        "",
        header,
        sep,
    ]

    for label, stats_per_comp in zip(labels, stats_list):
        for comp, st in zip(col_names, stats_per_comp):
            lines.append(
                f"{label:<20}  {comp:<12}  "
                f"{st['mae'] * scale:>10.4g}  "
                f"{st['rmse'] * scale:>10.4g}  "
                f"{st['mean'] * scale:>10.4g}  "
                f"{st['mad'] * scale:>10.4g}  "
                f"{st['q95'] * scale:>10.4g}  "
                f"{st['n']:>8d}"
            )
        lines.append("")

    with open(log_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logging.info(f"Metrics written to {log_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot error distributions between reference data and "
            "torchkrates-eval predictions."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Reference data file (.xyz, .extxyz, .h5, .hdf5)",
    )
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="FILE",
        help="Eval HDF5 file from torchkrates-eval. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="LABEL",
        help=(
            "Display label for the corresponding --eval file "
            "(positionally matched). Defaults to the file name."
        ),
    )
    parser.add_argument(
        "--property",
        required=True,
        choices=list(PROPERTY_META.keys()),
        help="Property to analyse.",
    )
    parser.add_argument(
        "--layout",
        choices=["overlay", "subplots"],
        default="overlay",
        help="'overlay': all models on same axes. "
        "'subplots': one row per model (default: overlay).",
    )
    parser.add_argument(
        "--output",
        default="errors.png",
        help="Output image path (default: errors.png).",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Write a metrics table to this text file.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    parser.add_argument(
        "--energy-key",
        default="REF_energy",
        help="Key for reference energy in data file (default: REF_energy).",
    )
    parser.add_argument(
        "--forces-key",
        default="REF_forces",
        help="Key for reference forces (default: REF_forces).",
    )
    parser.add_argument(
        "--stress-key",
        default="REF_stress",
        help="Key for reference stress (default: REF_stress).",
    )
    parser.add_argument(
        "--dipole-key",
        default="REF_dipoles",
        help="Key for reference dipole (default: REF_dipoles).",
    )
    parser.add_argument(
        "--hirshfeld-key",
        default="REF_hirsh_ratios",
        help=(
            "Key for reference Hirshfeld ratios "
            "(default: REF_hirsh_ratios)."
        ),
    )
    parser.add_argument(
        "--charges-key",
        default="REF_charges",
        help="Key for reference charges (default: REF_charges).",
    )

    args = parser.parse_args()

    # Resolve labels
    labels = args.label or []
    if len(labels) < len(args.eval):
        import os

        labels += [os.path.basename(p) for p in args.eval[len(labels) :]]

    # Build key map
    keys = {
        "energy_key": args.energy_key,
        "forces_key": args.forces_key,
        "stress_key": args.stress_key,
        "dipole_key": args.dipole_key,
        "hirshfeld_key": args.hirshfeld_key,
        "charges_key": args.charges_key,
    }

    # Load reference
    ref, n_atoms_list = load_reference(args.data, args.property, keys)

    # Load predictions and compute errors for each eval file
    errors_list = []
    stats_list = []
    meta = PROPERTY_META[args.property]

    for eval_path, label in zip(args.eval, labels):
        logging.info(f"Processing {label} ({eval_path})")
        pred = load_predictions(eval_path, args.property, n_atoms_list)

        if pred.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch for {label}: "
                f"predictions {pred.shape} vs reference {ref.shape}. "
                "Make sure --data matches the dataset used for evaluation."
            )

        errors = pred - ref
        if errors.ndim > 1 and len(meta["components"]) == 1:
            errors = errors.flatten()
        per_comp_stats = compute_per_component_stats(
            errors, meta["components"]
        )
        errors_list.append(errors)
        stats_list.append(per_comp_stats)
        logging.info(
            f"{label}: MAE={per_comp_stats[-1]['mae'] * meta['scale']:.4g} "
            f"{meta['unit']}, "
            f"RMSE={per_comp_stats[-1]['rmse'] * meta['scale']:.4g} "
            f"{meta['unit']}"
        )

    # Plot
    fig = build_plot(
        errors_list,
        stats_list,
        labels,
        args.property,
        args.layout,
        args.bins,
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    logging.info(f"Saved plot to {args.output}")

    # Optional metrics log
    if args.log:
        write_metrics_table(
            stats_list,
            labels,
            meta["components"],
            meta["unit"],
            meta["scale"],
            args.log,
        )


if __name__ == "__main__":
    main()
