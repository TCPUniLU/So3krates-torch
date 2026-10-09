"""Tests for the torchkrates-md-nvalchemi CLI helpers."""

import pytest

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")
pytest.importorskip("nvalchemiops", reason="nvalchemi-toolkit-ops missing")


def test_cli_module_importable_from_cli_package():
    from so3krates_torch.cli import run_md_nvalchemi

    assert callable(run_md_nvalchemi.main)
