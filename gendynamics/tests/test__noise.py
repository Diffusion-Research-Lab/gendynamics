"""Tests for noise and synthetic-data samplers."""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from gendynamics._noise import sample_checker, sample_exponential, sample_gaussian, sample_scaled_isotropic_alpha_stable, sample_scaled_scalar_alpha_stable, sample_spiral, sample_student_t, sample_unbalanced_highdim_alpha_stable_mixture, sample_unbalanced_highdim_gaussian_mixture, _orthonormal_embedding, _structured_mode_codebook


def _import_vendor_dlpm_class():
    """Import the vendored DLPM class directly for sampler-equivalence tests."""
    root = Path(__file__).resolve().parents[1] / "_vendor" / "DLPM"
    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        from dlpm.methods.dlpm import DLPM as VendorDLPM  # noqa: E402
    finally:
        sys.path[:] = [entry for entry in sys.path if entry != root_str]
    return VendorDLPM


def _assert_allclose_scalar(x: torch.Tensor, y: float, atol: float, rtol: float = 0.0):
    y_t = torch.tensor(y, device=x.device, dtype=x.dtype)
    assert torch.allclose(x, y_t, atol=atol, rtol=rtol), f"{x.item()} vs {y}"


def _cov(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / float(x.shape[0] - 1)


def test_sample_gaussian_mean_cov_identity_cpu():
    torch.manual_seed(0)
    n, d = 30000, 5
    x = sample_gaussian(n, d, device=torch.device("cpu"), dtype=torch.float64)
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    mean = x.mean(dim=0)
    cov = _cov(x)

    assert mean.abs().max().item() < 0.03

    diag = torch.diag(cov)
    assert (diag - 1.0).abs().max().item() < 0.05

    off = cov - torch.diag(diag)
    assert off.abs().max().item() < 0.03


def test_sample_student_t_mean_and_variance_when_finite():
    torch.manual_seed(1)
    n, d = 60000, 3
    nu = 7.0  # variance exists, Var = nu/(nu-2)
    x = sample_student_t(n, d, nu=nu, device=torch.device("cpu"), dtype=torch.float64)
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    mean = x.mean(dim=0)
    assert mean.abs().max().item() < 0.05

    var_emp = x.var(dim=0, unbiased=True)
    var_true = nu / (nu - 2.0)
    assert (var_emp - var_true).abs().max().item() < 0.12


def test_spiral_radius_moments_no_noise():
    torch.manual_seed(2)
    n = 40000
    turns = 3.0
    R = 4.0
    x = sample_spiral(
        n_samples=n,
        spiral_turns=turns,
        spiral_radius=R,
        spiral_noise=0.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert x.shape == (n, 2)
    assert torch.isfinite(x).all()

    r = torch.linalg.vector_norm(x, ord=2, dim=1)
    assert r.min().item() >= 0.0
    assert r.max().item() <= R + 1e-8

    # If r = R * U with U~Unif(0,1): E[r] = R/2, Var[r] = R^2/12
    _assert_allclose_scalar(r.mean(), R / 2.0, atol=0.03)
    _assert_allclose_scalar(r.var(unbiased=True), (R * R) / 12.0, atol=0.06)


def test_spiral_noise_increases_radius_variance():
    torch.manual_seed(3)
    r_tol = 1e-2
    a_tol = 5e-2
    n = 20000
    R = 4.0
    x0 = sample_spiral(n, spiral_radius=R, spiral_noise=0.0, device="cpu", dtype=torch.float64)
    x1 = sample_spiral(n, spiral_radius=R, spiral_noise=0.25, device="cpu", dtype=torch.float64)

    r0 = torch.linalg.vector_norm(x0, dim=1)
    r1 = torch.linalg.vector_norm(x1, dim=1)

    assert (1 + r_tol) * r1.var(unbiased=True).item() + a_tol > r0.var(unbiased=True).item()


def test_unbalanced_highdim_gaussian_mixture_exhibits_mode_imbalance():
    torch.manual_seed(5)
    n, d = 50_000, 20
    x = sample_unbalanced_highdim_gaussian_mixture(
        n,
        d,
        n_modes=12,
        rank=4,
        imbalance_tau=1.3,
        mean_scale=6.0,
        base_std=0.35,
        anisotropy=0.8,
        device="cpu",
        dtype=torch.float64,
    )
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    # Use the first embedded coordinate as a coarse mode proxy and verify a heavy head-tail imbalance.
    bins = torch.histc(x[:, 0].float(), bins=24)
    positive_bins = bins[bins > 0]
    assert len(positive_bins) >= 8
    assert (positive_bins.max() / positive_bins.min()).item() > 8.0


def test_unbalanced_highdim_alpha_stable_mixture_returns_heavy_tailed_samples():
    torch.manual_seed(6)
    np.random.seed(6)
    n, d = 10_000, 20
    x = sample_unbalanced_highdim_alpha_stable_mixture(
        n,
        d,
        alpha=1.6,
        n_modes=12,
        rank=4,
        imbalance_tau=1.3,
        mean_scale=6.0,
        base_scale=0.35,
        anisotropy=0.8,
        device="cpu",
        dtype=torch.float64,
    )
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    abs_values = x.abs().reshape(-1)
    assert torch.quantile(abs_values, 0.99) > 4.0 * torch.quantile(abs_values, 0.5)


def test_scaled_scalar_alpha_stable_shape_and_positivity():
    torch.manual_seed(7)
    n = 40000
    alpha = 1.7
    a = sample_scaled_scalar_alpha_stable(n, alpha=alpha, device="cpu", dtype=torch.float64)
    assert a.shape == (n, 1)
    assert torch.isfinite(a).all()
    assert (a > 0).all()


def test_scaled_scalar_alpha_stable_matches_vendor_dlpm_draw():
    n = 256
    alpha = 1.9
    VendorDLPM = _import_vendor_dlpm_class()
    vendor_dlpm = VendorDLPM(
        alpha=alpha,
        device="cpu",
        diffusion_steps=8,
        time_spacing="linear",
        isotropic=True,
        scale="scale_preserving",
    )

    np.random.seed(0)
    a_native = sample_scaled_scalar_alpha_stable(n, alpha=alpha, device="cpu", dtype=torch.float32)
    np.random.seed(0)
    a_vendor = vendor_dlpm.get_one_rv_faster_sampling((n,))

    assert a_native.shape == (n, 1)
    assert a_vendor.shape == (n,)
    assert torch.equal(a_native.reshape(-1), a_vendor)


def test_scaled_isotropic_alpha_stable_2d_direction_uniformity_and_symmetry():
    torch.manual_seed(8)
    n = 60000
    alpha = 1.7
    x = sample_scaled_isotropic_alpha_stable(n, dim=2, alpha=alpha, device="cpu", dtype=torch.float64)
    assert x.shape == (n, 2)
    assert torch.isfinite(x).all()

    # Direction should be uniform because X = sqrt(A) * G and G has uniform direction.
    theta = torch.atan2(x[:, 1], x[:, 0])
    c = torch.cos(theta).mean()
    s = torch.sin(theta).mean()
    assert abs(c.item()) < 0.015
    assert abs(s.item()) < 0.015

    # Symmetry: each coordinate should be ~50% positive
    frac_pos0 = (x[:, 0] > 0.0).to(torch.float64).mean().item()
    frac_pos1 = (x[:, 1] > 0.0).to(torch.float64).mean().item()
    assert abs(frac_pos0 - 0.5) < 0.015
    assert abs(frac_pos1 - 0.5) < 0.015


@pytest.mark.parametrize("alpha", [0.0, 2.1])
def test_input_validation_alpha_stable(alpha):
    with pytest.raises(ValueError):
        sample_scaled_scalar_alpha_stable(10, alpha=alpha, device="cpu")


def test_alpha_stable_alpha_eq_2_returns_constant_2():
    a = sample_scaled_scalar_alpha_stable(8, alpha=2.0, device="cpu", dtype=torch.float32)
    assert a.shape == (8, 1)
    assert (a == 2.0).all()


def test_sample_checker_shape_and_finiteness():
    torch.manual_seed(0)
    x = sample_checker(n_samples=500, device="cpu", dtype=torch.float32)
    assert x.shape == (500, 2)
    assert torch.isfinite(x).all()


def test_sample_exponential_valid_returns_positive():
    torch.manual_seed(0)
    x = sample_exponential(n_samples=200, dim=3, rate=2.0, device="cpu", dtype=torch.float32)
    assert x.shape == (200, 3)
    assert (x > 0).all()
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_sample_exponential_nonpositive_rate_raises(rate):
    with pytest.raises(ValueError, match="rate must be > 0"):
        sample_exponential(10, dim=2, rate=rate)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_modes": 1}, "n_modes"),
        ({"n_modes": 4, "imbalance_tau": -0.1}, "imbalance_tau"),
        ({"n_modes": 4, "mean_scale": 0.0}, "mean_scale"),
    ],
)
def test_highdim_gaussian_mixture_validates_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample_unbalanced_highdim_gaussian_mixture(10, dim=4, **kwargs)


def test_alpha_stable_highdim_mixture_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        sample_unbalanced_highdim_alpha_stable_mixture(10, dim=4, alpha=0.0)


def test_orthonormal_embedding_rank_exceeds_dim_raises():
    with pytest.raises(ValueError, match="rank"):
        _orthonormal_embedding(dim=3, rank=5, device=torch.device("cpu"), dtype=torch.float32)


def test_structured_mode_codebook_zero_rank_raises():
    with pytest.raises(ValueError, match="rank"):
        _structured_mode_codebook(n_modes=4, rank=0, device=torch.device("cpu"), dtype=torch.float32)
