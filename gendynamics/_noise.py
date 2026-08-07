"""Noise and synthetic-data sampling functions."""

import numpy as np
import scipy
import torch


def sample_scaled_scalar_alpha_stable(
    n_samples: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample positive scalar mixing coefficients for isotropic alpha-stable sampling."""
    alpha = float(alpha)

    if not (0.0 < alpha <= 2.0):
        raise ValueError(f"`alpha` must be in (0,2], got {alpha}.")
    if alpha == 2.0:
        return (2.0 * torch.ones((n_samples, 1), device=device, dtype=dtype))

    scale = 2.0 * np.cos(np.pi * alpha / 4.0) ** (2.0 / alpha)
    draws = scipy.stats.levy_stable.rvs(alpha / 2.0, 1.0, loc=0.0, scale=scale, size=n_samples)

    return torch.as_tensor(draws, device=device, dtype=dtype).unsqueeze(-1)


def sample_scaled_isotropic_alpha_stable(
    n_samples: int,
    dim: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample isotropic alpha-stable vectors via Gaussian scale mixing.

    Draws
        G ~ N(0, I) in R^dim
    and
        A from sample_scaled_scalar_alpha_stable(...),
    then returns
        X = sqrt(A) * G.
    """
    G = torch.randn(n_samples, dim, device=device, dtype=dtype)
    A = sample_scaled_scalar_alpha_stable(n_samples, alpha=alpha, device=device, dtype=dtype)
    return A.sqrt() * G


def sample_spiral(
    n_samples: int,
    spiral_turns: float = 3.0,
    spiral_radius: float = 4.0,
    spiral_noise: float = 0.15,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a 2d spirale cloud points.
    """
    u = torch.rand((n_samples,), device=device, dtype=dtype)
    theta = (2.0 * np.pi * spiral_turns) * u
    r = spiral_radius * u

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    pts = torch.stack([x, y], dim=1)

    if spiral_noise > 0.0:
        pts += spiral_noise * torch.randn_like(pts)

    return pts


def sample_checker(
    n_samples: int,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a 2d checker cloud points.
    """
    x1 = torch.rand(n_samples, device=device, dtype=dtype) * 4 - 2
    x2_ = torch.rand(n_samples, device=device, dtype=dtype)
    x2_ -= torch.randint(high=2, size=(n_samples, ), device=device, dtype=dtype) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    pts = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return pts.float()


def sample_student_t(
    n_samples: int,
    dim: int,
    nu: float,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a multivariate Student's t with identity scale and dof=nu.
    """
    z = torch.randn(n_samples, dim, device=device, dtype=dtype)
    s = torch.distributions.Gamma(nu / 2.0, 0.5).sample((n_samples,)).to(device=device, dtype=dtype)
    scale = torch.sqrt((s / nu).clamp_min(torch.finfo(dtype).tiny)).unsqueeze(1)
    return z / scale


def sample_exponential(
    n_samples: int,
    dim: int,
    rate: float = 1.0,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample from an exponential distribution Exp(rate).
    """
    if rate <= 0:
        raise ValueError("rate must be > 0.")

    return torch.empty(n_samples, dim, device=device, dtype=dtype).exponential_(rate)


def sample_gaussian(
    n_samples: int,
    dim: int,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a standard Gaussian N(0, I).
    """
    return torch.randn(n_samples, dim, device=device, dtype=dtype)


def _structured_mode_codebook(
    n_modes: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    structure_seed: int = 0,
) -> torch.Tensor:
    """Build one deterministic low-rank codebook for mixture centers."""
    if n_modes <= 0:
        raise ValueError("n_modes must be > 0.")
    if rank <= 0:
        raise ValueError("rank must be > 0.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(structure_seed))
    codebook = torch.randn(n_modes, rank, generator=generator, dtype=dtype)
    codebook = torch.nn.functional.normalize(codebook, dim=1)
    return codebook.to(device=device, dtype=dtype)


def _orthonormal_embedding(
    dim: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    structure_seed: int = 0,
) -> torch.Tensor:
    """Build one deterministic orthonormal embedding from R^rank to R^dim."""
    if dim <= 0:
        raise ValueError("dim must be > 0.")
    if rank <= 0 or rank > dim:
        raise ValueError(f"rank must lie in [1, dim], got rank={rank}, dim={dim}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(structure_seed) + 1)
    basis = torch.randn(dim, rank, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q.to(device=device, dtype=dtype)


def sample_unbalanced_highdim_gaussian_mixture(
    n_samples: int,
    dim: int,
    n_modes: int = 16,
    rank: int = 6,
    imbalance_tau: float = 1.2,
    mean_scale: float = 7.5,
    base_std: float = 0.55,
    anisotropy: float = 1.0,
    structure_seed: int = 0,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a high-dimensional Gaussian mixture with many imbalanced modes.

    The mode centers live in a low-rank latent subspace and are embedded into the
    ambient space. Mixture weights follow a power-law decay to make rare modes easy
    to collapse under limited model capacity.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if dim <= 0:
        raise ValueError("dim must be > 0.")
    if n_modes <= 1:
        raise ValueError("n_modes must be > 1.")
    if imbalance_tau < 0.0:
        raise ValueError("imbalance_tau must be >= 0.")
    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if base_std <= 0.0:
        raise ValueError("base_std must be > 0.")
    if anisotropy < 0.0:
        raise ValueError("anisotropy must be >= 0.")

    centers, directions, mode_index = _unbalanced_highdim_mixture_structure(
        n_samples=n_samples,
        dim=dim,
        n_modes=n_modes,
        rank=rank,
        imbalance_tau=imbalance_tau,
        mean_scale=mean_scale,
        structure_seed=structure_seed,
        device=device,
        dtype=dtype,
    )

    noise_iso = torch.randn(n_samples, dim, device=device, dtype=dtype)
    noise_axis = torch.randn(n_samples, 1, device=device, dtype=dtype) * directions[mode_index]
    noise = noise_iso + anisotropy * noise_axis

    return centers[mode_index] + base_std * noise


def _unbalanced_highdim_mixture_structure(
    n_samples: int,
    dim: int,
    n_modes: int,
    rank: int,
    imbalance_tau: float,
    mean_scale: float,
    structure_seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shared centers, directions, and sampled mode ids for high-dimensional mixtures."""
    rank = min(int(rank), int(dim))
    embedding = _orthonormal_embedding(dim, rank, device=device, dtype=dtype, structure_seed=structure_seed)
    codebook = _structured_mode_codebook(n_modes, rank, device=device, dtype=dtype, structure_seed=structure_seed)
    centers = mean_scale * (codebook @ embedding.T)
    directions = torch.nn.functional.normalize(centers, dim=1)

    weights = torch.arange(1, n_modes + 1, device=device, dtype=dtype).pow(-imbalance_tau)
    weights = weights / weights.sum()
    mode_index = torch.multinomial(weights, n_samples, replacement=True)
    return centers, directions, mode_index


def sample_unbalanced_highdim_alpha_stable_mixture(
    n_samples: int,
    dim: int,
    alpha: float = 1.7,
    n_modes: int = 16,
    rank: int = 6,
    imbalance_tau: float = 1.2,
    mean_scale: float = 7.5,
    base_scale: float = 0.55,
    anisotropy: float = 1.0,
    structure_seed: int = 0,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample an imbalanced high-dimensional mixture with alpha-stable local noise.

    This uses the same low-rank centers and power-law mode weights as
    ``sample_unbalanced_highdim_gaussian_mixture`` but replaces each component's
    Gaussian noise with symmetric alpha-stable noise.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if dim <= 0:
        raise ValueError("dim must be > 0.")
    if n_modes <= 1:
        raise ValueError("n_modes must be > 1.")
    if not (0.0 < float(alpha) <= 2.0):
        raise ValueError(f"`alpha` must be in (0,2], got {alpha}.")
    if imbalance_tau < 0.0:
        raise ValueError("imbalance_tau must be >= 0.")
    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if base_scale <= 0.0:
        raise ValueError("base_scale must be > 0.")
    if anisotropy < 0.0:
        raise ValueError("anisotropy must be >= 0.")

    centers, directions, mode_index = _unbalanced_highdim_mixture_structure(
        n_samples=n_samples,
        dim=dim,
        n_modes=n_modes,
        rank=rank,
        imbalance_tau=imbalance_tau,
        mean_scale=mean_scale,
        structure_seed=structure_seed,
        device=device,
        dtype=dtype,
    )

    noise_iso = sample_scaled_isotropic_alpha_stable(n_samples, dim=dim, alpha=alpha, device=device, dtype=dtype)
    noise_axis = sample_scaled_isotropic_alpha_stable(n_samples, dim=1, alpha=alpha, device=device, dtype=dtype) * directions[mode_index]
    noise = noise_iso + anisotropy * noise_axis

    return centers[mode_index] + base_scale * noise
