"""Tests for schedule helpers."""

import math
import pytest
import torch
from gendynamics._schedules import cosine_schedule
from .utils import _devices


@pytest.mark.parametrize("fdtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("idtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_schedule_shapes_dtypes_and_devices(fdtype, idtype, device):
    n_steps = 37
    alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(
        n_steps=n_steps, device=device, fdtype=fdtype, idtype=idtype, s=0.008
    )

    assert alpha_bar.shape == (n_steps,)
    assert alphas.shape == (n_steps,)
    assert betas.shape == (n_steps,)
    assert sqrt_post_var.shape == (n_steps,)

    assert alpha_bar.dtype == fdtype
    assert alphas.dtype == fdtype
    assert betas.dtype == fdtype
    assert sqrt_post_var.dtype == fdtype

    assert alpha_bar.device.type == device.type
    assert alphas.device.type == device.type
    assert betas.device.type == device.type
    assert sqrt_post_var.device.type == device.type


@pytest.mark.parametrize("fdtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("idtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_schedule_matches_definition(fdtype, idtype, device):
    n_steps = 123
    s = 0.01
    eps = 1e-8
    beta_max = 0.999
    alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(
        n_steps=n_steps, device=device, fdtype=fdtype, idtype=idtype, s=s, eps=eps, beta_max=beta_max
    )

    t = torch.arange(0, n_steps + 1, device=device, dtype=idtype)
    ref_alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2).clamp(min=eps)
    ref_alpha_bar = (ref_alpha_bar / ref_alpha_bar[0]).to(device=device, dtype=fdtype)
    ref_alphas = (ref_alpha_bar[1:] / ref_alpha_bar[:-1]).clamp(min=eps)
    ref_betas = (1.0 - ref_alphas).clamp(min=eps, max=beta_max)
    ref_alphas = 1.0 - ref_betas
    ref_alpha_bar = torch.cumprod(ref_alphas, dim=0)
    ref_alpha_bar_prev = torch.cat([torch.ones(1, device=device, dtype=fdtype), ref_alpha_bar[:-1]])
    ref_sqrt_post_var = torch.sqrt(ref_betas * (1.0 - ref_alpha_bar_prev) / (1.0 - ref_alpha_bar))

    assert torch.allclose(alpha_bar, ref_alpha_bar, atol=0.0, rtol=0.0)
    assert torch.allclose(alphas, ref_alphas, atol=0.0, rtol=0.0)
    assert torch.allclose(betas, ref_betas, atol=0.0, rtol=0.0)
    assert torch.allclose(sqrt_post_var, ref_sqrt_post_var, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("fdtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("idtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_schedule_identities_and_bounds(fdtype, idtype, device):
    n_steps = 200
    alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(
        n_steps=n_steps, device=device, fdtype=fdtype, idtype=idtype, s=0.008
    )

    assert torch.isfinite(alpha_bar).all()
    assert torch.isfinite(alphas).all()
    assert torch.isfinite(betas).all()
    assert torch.isfinite(sqrt_post_var).all()

    assert torch.allclose(alphas, 1.0 - betas, atol=1e-7, rtol=1e-7)
    assert (alpha_bar <= 1.0).all()
    assert (alpha_bar > 0.0).all()
    assert (alphas > 0.0).all()
    assert (alphas <= 1.0).all()
    assert (betas >= 0.0).all()
    assert (betas < 1.0).all()
    assert (sqrt_post_var >= 0.0).all()

    diffs = alpha_bar[1:] - alpha_bar[:-1]
    assert (diffs <= 1e-12).all()


def test_cosine_schedule_caps_beta():
    _, _, betas, _ = cosine_schedule(
        n_steps=100,
        device=torch.device("cpu"),
        fdtype=torch.float32,
        idtype=torch.int64,
        beta_max=0.999,
    )

    assert betas.max() <= torch.tensor(0.999, dtype=betas.dtype)


def test_cosine_schedule_n_steps_zero_returns_empty_tensors():
    alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(
        n_steps=0,
        device=torch.device("cpu"),
        fdtype=torch.float32,
        idtype=torch.int64,
        s=0.008,
    )

    assert alpha_bar.shape == (0,)
    assert alphas.shape == (0,)
    assert betas.shape == (0,)
    assert sqrt_post_var.shape == (0,)
