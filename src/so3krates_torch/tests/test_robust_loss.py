"""Torch port of Barron's general / adaptive robust loss."""

import math

import numpy as np
import pytest
import torch

from so3krates_torch.modules.robust_loss import Distribution, general_loss

X = torch.linspace(-5.0, 5.0, 41, dtype=torch.float64)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_general_loss_alpha_2_is_half_l2(c):
    assert torch.allclose(general_loss(X, 2.0, c), 0.5 * (X / c) ** 2)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_general_loss_alpha_1_is_charbonnier(c):
    expected = torch.sqrt((X / c) ** 2 + 1) - 1
    assert torch.allclose(general_loss(X, 1.0, c), expected, atol=1e-6)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_general_loss_alpha_0_is_cauchy(c):
    expected = torch.log(0.5 * (X / c) ** 2 + 1)
    assert torch.allclose(general_loss(X, 0.0, c), expected)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_nll_alpha_2_is_gaussian(c):
    nll = Distribution().nllfun(
        X, torch.tensor(2.0, dtype=X.dtype), torch.tensor(c, dtype=X.dtype)
    )
    expected = 0.5 * (X / c) ** 2 + math.log(c) + 0.5 * math.log(2 * math.pi)
    assert torch.allclose(nll, expected, atol=1e-5)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_nll_alpha_0_is_cauchy(c):
    nll = Distribution().nllfun(
        X, torch.tensor(0.0, dtype=X.dtype), torch.tensor(c, dtype=X.dtype)
    )
    expected = (
        torch.log(0.5 * (X / c) ** 2 + 1)
        + math.log(c)
        + math.log(math.pi * math.sqrt(2))
    )
    assert torch.allclose(nll, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nll_finite_at_extremes(dtype):
    dist = Distribution()
    for a in (0.0, 1e-4, 1.0, 2.0 - 1e-4, 2.0):
        for c in (1e-3, 1.0, 10.0):
            x = torch.tensor(
                [-1e4, -3.0, 0.0, 1e-8, 3.0, 1e4],
                dtype=dtype,
                requires_grad=True,
            )
            alpha = torch.tensor(a, dtype=dtype, requires_grad=True)
            scale = torch.tensor(c, dtype=dtype, requires_grad=True)
            nll = dist.nllfun(x, alpha, scale)
            assert torch.isfinite(nll).all(), (a, c)
            nll.sum().backward()
            for t in (x, alpha, scale):
                assert torch.isfinite(t.grad).all(), (a, c)


def test_general_loss_grad_finite_at_special_alphas():
    for a in (0.0, 1.0, 2.0):
        x = X.clone().requires_grad_(True)
        general_loss(x, a, 1.0).sum().backward()
        assert torch.isfinite(x.grad).all()


def test_matches_jax_robust_loss(jax_so3lr):
    import jax

    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        jdist = jax_so3lr.distribution.Distribution()
        tdist = Distribution()
        x = np.linspace(-20, 20, 81)
        for a in (0.0, 0.3, 1.0, 1.5, 1.99, 2.0):
            for c in (0.05, 1.0, 3.0):
                ref_rho = np.asarray(
                    jax_so3lr.general.lossfun(jnp.asarray(x), a, c)
                )
                ref_nll = np.asarray(
                    jdist.nllfun(jnp.asarray(x), jnp.asarray(a), c)
                )
                xt = torch.tensor(x)
                rho = general_loss(xt, a, c).numpy()
                nll = tdist.nllfun(
                    xt, torch.tensor(a, dtype=xt.dtype), torch.tensor(c)
                ).numpy()
                np.testing.assert_allclose(rho, ref_rho, rtol=1e-6, atol=1e-9)
                np.testing.assert_allclose(nll, ref_nll, rtol=1e-6, atol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_partition_spline_shipped():
    from importlib import resources

    path = resources.files("so3krates_torch.modules.robust_loss").joinpath(
        "data/partition_spline.npz"
    )
    with resources.as_file(path) as p, np.load(p) as f:
        assert {"x_scale", "values", "tangents"} <= set(f.files)


def test_wheel_ships_license_texts(tmp_path):
    """Apache-2.0 §4(a): the robust-loss code must ship with its license;
    the CC BY-NC text for the tauri weights likewise."""
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    import shutil

    # Build from a copy so setuptools' build/ and *.egg-info never land
    # in the working tree (weights skipped: irrelevant to licensing).
    repo = Path(__file__).resolve().parents[3]
    src = tmp_path / "src_copy"
    src.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE.md", "NOTICE.md"):
        shutil.copy(repo / name, src / name)
    shutil.copytree(repo / "licenses", src / "licenses")
    shutil.copytree(
        repo / "src",
        src / "src",
        ignore=shutil.ignore_patterns("*.pt", "__pycache__", "*.egg-info"),
    )
    wheels = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-q",
            "-w",
            str(wheels),
            str(src),
        ],
        check=True,
    )
    (wheel,) = wheels.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for required in (
        "LICENSE.ROBUST-LOSS-APACHE-2.0",
        "LICENSE.CC-BY-NC-4.0",
        "LICENSE.MACE-MIT",
        "LICENSE.PYG-MIT",
        "NOTICE.md",
    ):
        assert any(n.endswith(required) for n in names), required
