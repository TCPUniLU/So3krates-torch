"""Validation metrics for per-atom dispersion ratio labels.

Missing labels (NaN, or zero structure weight in old preprocessed
files) must not count as errors.
"""

import logging

import torch

from so3krates_torch.tools.eval import ModelEval
from so3krates_torch.tools.torch_geometric import Batch


def _batch(hirsh, c6, hirsh_w=(1.0, 1.0), c6_w=(1.0, 1.0)):
    return Batch(
        ptr=torch.tensor([0, 2, 4]),
        batch=torch.tensor([0, 0, 1, 1]),
        weight=torch.tensor([1.0, 1.0]),
        hirshfeld_ratios=torch.tensor(hirsh),
        hirshfeld_ratios_weight=torch.tensor(hirsh_w),
        c6_ratios=torch.tensor(c6),
        c6_ratios_weight=torch.tensor(c6_w),
    )


def _output(value=1.0):
    return {
        "hirshfeld_ratios": torch.full((4,), value),
        "c6_ratios": torch.full((4,), value),
    }


def test_eval_ratio_metrics_ignore_missing_labels():
    nan = float("nan")
    metric = ModelEval()
    # graph 0 labelled (errors 0.5), graph 1 missing
    metric.update(
        _batch([1.5, 1.5, nan, nan], [2.0, 2.0, nan, nan]), _output()
    )
    # zero-filled label with zero weight (old preprocessed files)
    metric.update(
        _batch(
            [0.0, 0.0, 1.5, 1.5], [0.0, 0.0, 2.0, 2.0], (0.0, 1.0), (0.0, 1.0)
        ),
        _output(),
    )
    aux = metric.compute()
    assert abs(aux["mae_hirshfeld_ratios"] - 0.5) < 1e-6
    assert abs(aux["mae_c6_ratios"] - 1.0) < 1e-6
    assert abs(aux["rmse_c6_ratios"] - 1.0) < 1e-6


def test_eval_ratio_metrics_absent_when_unlabelled():
    nan = float("nan")
    metric = ModelEval()
    metric.update(_batch([nan] * 4, [nan] * 4), _output())
    aux = metric.compute()
    assert "mae_c6_ratios" not in aux
    assert "mae_hirshfeld_ratios" not in aux


def test_valid_err_log_reports_c6(caplog):
    from so3krates_torch.tools.train import valid_err_log

    class _Logger:
        def log(self, _):
            pass

    with caplog.at_level(logging.INFO):
        valid_err_log(
            0.1,
            {"mae_e_per_atom": 0.001, "mae_f": 0.01, "mae_c6_ratios": 0.0123},
            _Logger(),
            "PerAtomMAE",
            epoch=3,
        )
    assert "MAE_c6_ratios=  0.0123" in caplog.text


def test_valid_err_log_hirshfeld_table_without_hirshfeld_metric(caplog):
    """Validation set without Hirshfeld labels: metric key is absent;
    the Hirshfeld error tables must not crash."""
    from so3krates_torch.tools.train import valid_err_log

    class _Logger:
        def log(self, _):
            pass

    base = {
        "mae_e_per_atom": 0.001,
        "mae_f": 0.01,
        "mae_dipole": 0.01,
        "mae_charges": 0.01,
    }
    with caplog.at_level(logging.INFO):
        for table in (
            "EnergyForceHirshfeldMAE",
            "EnergyForceDipoleHirshfeldMAE",
            "EnergyForceChargesHirshfeldMAE",
        ):
            valid_err_log(0.1, dict(base), _Logger(), table, epoch=1)
    assert caplog.text.count("MAE_hirshfeld_ratios=n/a") == 3
