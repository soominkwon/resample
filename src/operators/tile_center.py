import torch
import torch.nn.functional as F


def _make_center_grid(H, W, s, device):
    """Build a normalized grid for grid_sample with centers at each s×s tile.

    Returns:
        (1, Hs, Ws, 2) in [-1, 1] normed coords, where Hs = H // s, Ws = W // s.
    """
    assert H % s == 0 and W % s == 0, "H and W must be divisible by s"
    Hs, Ws = H // s, W // s
    cy = torch.arange(Hs, device=device) * s + (s - 1) / 2.0
    cx = torch.arange(Ws, device=device) * s + (s - 1) / 2.0
    yy, xx = torch.meshgrid(cy, cx, indexing="ij")
    grid_y = (yy / (H - 1)) * 2 - 1
    grid_x = (xx / (W - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return grid


def A(x, s):
    """Forward measurement: sample geometric centers of s×s tiles.

    Args:
        x: Tensor of shape (B, C, H, W) in [-1, 1].
        s: Tile size.

    Returns:
        Tensor of shape (B, C, H/s, W/s).
    """
    B, C, H, W = x.shape
    grid = _make_center_grid(H, W, s, x.device).repeat(B, 1, 1, 1)
    y = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return y


def AT(y, s, H, W):
    """Adjoint: back-project each measurement value uniformly to its tile.

    Args:
        y: Tensor of shape (B, C, H/s, W/s).
        s: Tile size.
        H, W: Spatial dims of desired output.

    Returns:
        Back-projected tensor of shape (B, C, H, W).
    """
    B, C, Hs, Ws = y.shape
    assert Hs * s == H and Ws * s == W, "H/W must match y and s"
    out = torch.zeros(B, C, H, W, device=y.device, dtype=y.dtype)
    for i in range(Hs):
        for j in range(Ws):
            out[:, :, i * s : (i + 1) * s, j * s : (j + 1) * s] = y[:, :, i : i + 1, j : j + 1]
    return out
