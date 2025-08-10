import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.operators.tile_center import A, AT


def test_A_shapes():
    x = torch.randn(2, 3, 256, 256)
    for s in [2, 4, 8]:
        y = A(x, s)
        assert y.shape == (2, 3, 256 // s, 256 // s)


def test_roundtrip_sanity():
    x = torch.randn(1, 1, 64, 64)
    for s in [2, 4, 8]:
        y = A(x, s)
        x_bp = AT(y, s, 64, 64)
        assert x_bp.shape == x.shape
        assert torch.isfinite(x_bp).all()
