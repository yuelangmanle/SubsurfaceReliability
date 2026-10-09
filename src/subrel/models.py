"""Neural solver architectures evaluated in the reliability benchmark.

Implements standard baseline configurations for three solver families:
  * UNet  -- Convolutional encoder-decoder regression surrogate on (ln K -> h)
  * FNO2d -- 2D Fourier Neural Operator (Li et al., 2021) with spectral modes
  * PINN  -- Per-instance coordinate MLP trained on the Darcy PDE residual with
             exact Dirichlet boundary enforcement and soft no-flow penalties

Grid-based models take standardized channels [ln K, x, y] and output head h.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- U-Net -------
class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        g = min(8, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin: int = 3, width: int = 32, depth: int = 4):
        super().__init__()
        chans = [width * 2**i for i in range(depth)]
        self.inc = _ConvBlock(cin, chans[0])
        self.downs = nn.ModuleList(
            [_ConvBlock(chans[i], chans[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in range(depth - 1)]
        )
        self.upblocks = nn.ModuleList(
            [_ConvBlock(2 * chans[i], chans[i]) for i in range(depth - 1)]
        )
        self.out = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.avg_pool2d(skips[-1], 2)))
        h = skips[-1]
        for i in reversed(range(len(self.ups))):
            h = self.ups[i](h)
            h = self.upblocks[i](torch.cat([h, skips[i]], dim=1))
        return self.out(h)


# ---------------------------------------------------------------- FNO ---------
class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes: int):
        super().__init__()
        self.cin, self.cout, self.modes = cin, cout, modes
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        # MPS has no complex FFT: run the spectral path on CPU
        dev = x.device
        xf = torch.fft.rfft2(x.float().cpu())
        out = torch.zeros(b, self.cout, h, w // 2 + 1, dtype=torch.cfloat)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1.cpu())
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2.cpu())
        return torch.fft.irfft2(out, s=(h, w)).to(dev)


class FNO2d(nn.Module):
    def __init__(self, cin: int = 3, width: int = 32, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(cin, width)
        self.specs = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):                       # x: (b, cin, h, w)
        h = self.lift(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for s, w in zip(self.specs, self.ws):
            h = F.gelu(s(h) + w(h))
        return self.proj(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------- PINN --------
class PINN(nn.Module):
    """Vanilla tanh MLP in (x, y) with hard Dirichlet left/right:

        h(x, y) = (1 - x) + x (1 - x) N(x, y)
    """

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers, cin = [], 2
        for _ in range(depth):
            lin = nn.Linear(cin, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            cin = width
        layers.append(nn.Linear(cin, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):                      # xy: (m, 2)
        x = xy[:, :1]
        return (1.0 - x) + x * (1.0 - x) * self.net(xy)


def pinn_residual(model: PINN, xy: torch.Tensor, Y: torch.Tensor, Yx: torch.Tensor, Yy: torch.Tensor,
                  unnormalized: bool = False):
    """PDE residual of -div(K grad h)=0 with K=exp(Y):

        div(K grad h) = K ( grad Y . grad h + lap h )

    Y, Yx, Yy are the log-conductivity and its gradient at the points xy.
    NB: Y* are reshaped to (m, 1) -- broadcasting a 1-D (m,) field against the
    (m, 1) gradient components would silently produce an (m, m) matrix.
    Returns (residual, hx, hy) at the points.
    """
    xy = xy.detach().clone().requires_grad_(True)   # in-place rearm: the
    # forward below must build a graph even if a previous call ran under
    # no_grad with this tensor (RAR resampling path)
    with torch.enable_grad():   # resampling scores points under a no_grad
        h = model(xy)           # caller; scoring itself needs no graph, but
        # the derivative calls below must run in graph mode
        g = torch.autograd.grad(h.sum(), xy, create_graph=True)[0]
        hx, hy = g[:, :1], g[:, 1:]
        hxx = torch.autograd.grad(hx.sum(), xy, create_graph=True)[0][:, :1]
        hyy = torch.autograd.grad(hy.sum(), xy, create_graph=True)[0][:, 1:]
    res = Yx.reshape(-1, 1) * hx + Yy.reshape(-1, 1) * hy + hxx + hyy  # /K: same zero set
    if unnormalized:
        # multiply back by K = exp(Y): same solution set, different objective
        res = torch.exp(Y.reshape(-1, 1)) * res
    return res, hx, hy


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
