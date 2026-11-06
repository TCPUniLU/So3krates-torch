"""Skip marker for tests that need the v2 checkpoints.

The v2 checkpoints are private and git-ignored
(``pretrained/so3lr/v2/``), so they are absent in CI and in fresh clones.
"""

import importlib.resources as resources

import pytest

_V2_S_SETTINGS = (
    resources.files("so3krates_torch.pretrained.so3lr")
    / "v2"
    / "so3lr-s_settings.yaml"
)

requires_v2_s = pytest.mark.skipif(
    not _V2_S_SETTINGS.is_file(),
    reason="v2 checkpoints are private and not bundled (git-ignored)",
)
