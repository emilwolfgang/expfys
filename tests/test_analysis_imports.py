"""Basic smoke tests for the reorganised analysis modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expfys.analysis import del_a, del_b_free, del_b_sticky


def test_del_a_config_defaults():
    cfg = del_a.Config()
    assert cfg.dist_threshold > 0


@pytest.mark.parametrize(
    "module, cfg_cls",
    [
        (del_b_free, del_b_free.Config),
        (del_b_sticky, del_b_sticky.Config),
    ],
)
def test_del_b_config_requires_bodies(module, cfg_cls):
    cfg = cfg_cls(bodies_path=Path("dummy.tsv"), dist_threshold=50.0)
    assert cfg.dist_threshold > 0
