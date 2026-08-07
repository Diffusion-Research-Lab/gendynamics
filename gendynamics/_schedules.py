"""Inference timestep schedules for flow samplers."""

import math
from typing import Any
import torch


def _check_n_steps(n_steps: int) -> int:
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    return n_steps


@torch.no_grad()
def cosine_schedule(
    n_steps: int,
    device: torch.device,
    fdtype: torch.dtype,
    idtype: torch.dtype,
    s: float = 0.008,
    eps: float = 1e-8,
    beta_max: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the cosine DDPM variance schedule."""
    t = torch.arange(0, n_steps + 1, device=device, dtype=idtype)

    alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2).clamp(min=eps)
    alpha_bar = (alpha_bar / alpha_bar[0]).to(device=device, dtype=fdtype)

    alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=eps)
    betas = (1.0 - alphas).clamp(min=eps, max=beta_max)
    alphas = 1.0 - betas

    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device, dtype=fdtype), alpha_bar[:-1]])
    sqrt_post_var = torch.sqrt(betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(eps))

    return alpha_bar, alphas, betas, sqrt_post_var


def linear_timesteps(
    n_steps: int,
    t_min: float,
    t_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an evenly spaced grid with `n_steps` integration intervals."""
    n_steps = _check_n_steps(n_steps)
    return torch.linspace(float(t_min), float(t_max), n_steps + 1, device=device, dtype=dtype)


def quadratic_timesteps(
    n_steps: int,
    t_min: float,
    t_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a quadratic grid with denser points near `t_min`."""
    n_steps = _check_n_steps(n_steps)
    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    return float(t_min) + (float(t_max) - float(t_min)) * u.square()


def flux_shifted_timesteps(
    n_steps: int,
    t_min: float,
    t_max: float,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a FLUX-style shifted grid as a function of image sequence length."""
    n_steps = _check_n_steps(n_steps)
    image_seq_len = int(image_seq_len)
    base_image_seq_len = int(base_image_seq_len)
    max_image_seq_len = int(max_image_seq_len)

    if image_seq_len <= 0:
        raise ValueError(f"image_seq_len must be positive, got {image_seq_len}.")
    if max_image_seq_len <= base_image_seq_len:
        raise ValueError("max_image_seq_len must be greater than base_image_seq_len.")

    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype

    ratio = (float(image_seq_len) - float(base_image_seq_len)) / (float(max_image_seq_len) - float(base_image_seq_len))
    ratio = min(max(ratio, 0.0), 1.0)

    mu = float(base_shift) + ratio * (float(max_shift) - float(base_shift))
    exp_mu = math.exp(mu)

    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    shifted = (exp_mu * u) / (1.0 + (exp_mu - 1.0) * u)
    return float(t_min) + (float(t_max) - float(t_min)) * shifted


def build_flow_timesteps(
    schedule: str,
    n_steps: int,
    t_min: float,
    t_max: float,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs: Any,
) -> torch.Tensor:
    """Build a named flow inference timestep schedule."""
    name = schedule

    if name == "linear":
        if kwargs:
            raise TypeError(f"Unexpected linear schedule kwargs: {sorted(kwargs)}.")
        return linear_timesteps(n_steps, t_min, t_max, device, dtype)

    if name == "quadratic":
        if kwargs:
            raise TypeError(f"Unexpected quadratic schedule kwargs: {sorted(kwargs)}.")
        return quadratic_timesteps(n_steps, t_min, t_max, device, dtype)

    if name == "flux_shifted":
        return flux_shifted_timesteps(n_steps, t_min, t_max, device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown flow timestep schedule {schedule!r}.")
