"""Conformer block with rotary self-attention. Every norm is per frame, so padded frames never
change the result for real frames."""
import torch
from torch import nn
import torch.nn.functional as F


class RotaryAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("dim / heads must be an even integer")
        self.heads, self.dropout = heads, dropout
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // heads, 2).float() / (dim // heads)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate(self, x):
        angles = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)[:, None] * self.inv_freq
        cos, sin = angles.cos(), angles.sin()
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x, pad_mask=None):
        b, t, d = x.shape
        split = lambda y: y.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = self._rotate(split(self.q(x))), self._rotate(split(self.k(x))), split(self.v(x))
        mask = None if pad_mask is None else ~pad_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ConvModule(nn.Module):
    def __init__(self, dim, kernel, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.post_norm = nn.LayerNorm(dim)
        self.project = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        y = F.glu(self.expand(self.norm(x).transpose(1, 2)), dim=1)
        if pad_mask is not None:
            y = y.masked_fill(pad_mask[:, None, :], 0.0)
        y = F.silu(self.post_norm(self.depthwise(y).transpose(1, 2)))
        return self.dropout(self.project(y.transpose(1, 2)).transpose(1, 2))


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up, self.down = nn.Linear(dim, mult * dim), nn.Linear(mult * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(self.dropout(F.silu(self.up(self.norm(x))))))


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, conv_kernel=15, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_mult, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RotaryAttention(dim, heads, dropout)
        self.conv = ConvModule(dim, conv_kernel, dropout)
        self.ff2 = FeedForward(dim, ff_mult, dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.dropout(self.attn(self.attn_norm(x), pad_mask))
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)
