"""Diffusion module indexing tests."""

import pytest
import torch
from gendynamics.diffusion import DLPMEps
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def _make_dlpmeps(n_steps: int = 7, device="cpu", dtype=torch.float32) -> DLPMEps:
    return DLPMEps(net=_ZeroNet(), dim=2, n_steps=n_steps, device=device, fdtype=dtype)


def _assert_finite_scalar(x: torch.Tensor):
    assert x.ndim == 0
    assert torch.isfinite(x).item()


def test_dlpmeps_loss_samples_vendor_time_range(monkeypatch):
    model = _make_dlpmeps(n_steps=7)
    x = torch.randn(5, 2, dtype=torch.float32)

    def _patched_randint(low, high, size, device=None, **kwargs):
        assert low == 1
        assert high == model._n_steps
        return torch.full(size, high - 1, device=device, dtype=torch.int64)

    monkeypatch.setattr(torch, "randint", _patched_randint)
    loss = model.loss(x)
    _assert_finite_scalar(loss)


def test_dlpmeps_sample_uses_vendor_terminal_sigma_index(monkeypatch):
    n_steps = 4
    n_samples = 3
    dim = 2
    model = _make_dlpmeps(n_steps=n_steps)

    model._sigma_1_t = torch.tensor([0.0, 0.0, 0.0, 7.0, 99.0], dtype=torch.float32)
    model._draw_A = lambda n: torch.ones(n, 1, device=model._device, dtype=model._fdtype)

    def _patched_randn(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _patched_randn)

    def _neutral_reverse(Sigma_1_t, t):
        sigma_hat = torch.zeros(n_samples, device=model._device, dtype=model._fdtype)
        gamma_t = torch.tensor(1.0, device=model._device, dtype=model._fdtype)
        Gamma_t = torch.zeros(n_samples, device=model._device, dtype=model._fdtype)
        return sigma_hat, gamma_t, Gamma_t

    model._g_Sigma_hat_Gamma = _neutral_reverse

    out = model.sample(n_samples=n_samples)
    expected = torch.full((n_samples, dim), 7.0, dtype=model._fdtype, device=model._device)
    assert torch.allclose(out, expected, atol=2e-3, rtol=0.0)


def test_dlpmeps_sample_requests_final_state_only(monkeypatch):
    model = _make_dlpmeps(n_steps=4)
    seen = {}

    def _patched_sample(n_samples, return_trajectory=True):
        seen["n_samples"] = n_samples
        seen["return_trajectory"] = return_trajectory
        return torch.zeros(n_samples, 2), None

    monkeypatch.setattr(model, "_sample", _patched_sample)

    out = model.sample(3)

    assert out.shape == (3, 2)
    assert seen == {"n_samples": 3, "return_trajectory": False}


def test_dlpmeps_sample_trajectory_is_optional():
    n_steps = 4
    n_samples = 2
    model = _make_dlpmeps(n_steps=n_steps)

    torch.manual_seed(0)
    out, trajectory = model._sample(n_samples, return_trajectory=False)

    assert out.shape == (n_samples, 2)
    assert trajectory is None

    torch.manual_seed(0)
    out_with_trajectory, trajectory = model._sample(n_samples, return_trajectory=True)

    assert out_with_trajectory.shape == (n_samples, 2)
    assert len(trajectory) == n_steps
    assert all(x.shape == (n_samples, 2) for x in trajectory)


def test_dlpmeps_accepts_image_shaped_batches():
    model = DLPMEps(net=_ZeroNet(), dim=(1, 4, 4), n_steps=4, device="cpu", fdtype=torch.float32)
    x = torch.randn(3, 1, 4, 4)

    loss = model.loss(x, t=0.5)

    _assert_finite_scalar(loss)
    assert model.sample(2).shape == (2, 1, 4, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("t", [3, 0.5])
def test_dlpmeps_loss_accepts_common_time_formats(t, device, dtype):
    model = _make_dlpmeps(n_steps=7, device=device, dtype=dtype)
    x = torch.randn(5, 2, dtype=dtype, device=device)
    _assert_finite_scalar(model.loss(x, t=t))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_dlpmeps_loss_rejects_invalid_t(device, dtype):
    model = _make_dlpmeps(n_steps=7, device=device, dtype=dtype)
    x = torch.randn(5, 2, dtype=dtype, device=device)

    with pytest.raises(ValueError):
        model.loss(x, t=0)


def test_dlpmeps_sample_accepts_native_sampler():
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="native")

    out = model.sample(2)

    assert out.shape == (2, 2)


def test_dlpmeps_uses_constructor_sampler_by_default():
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="native")

    out = model.sample(2)

    assert out.shape == (2, 2)


def test_dlpmeps_sample_rejects_non_native_sampler():
    with pytest.raises(ValueError, match="native"):
        DLPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="heun")
