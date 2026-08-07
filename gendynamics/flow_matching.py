"""Flow-matching models."""

import numpy as np
import torch
import ot
from ._abs import GaussianFlowAbstract
from ._schedules import build_flow_timesteps
from ._solvers import sample_flow, sample_flow_adaptive_heun


class GaussianFlowLinear(GaussianFlowAbstract):
    """Gaussian source flow with a linear path (Flow Matching)."""
    _family = "flow"
    _loss_tag = "mse"

    def _precompute_loss(self, x, z, t=None):
        """Build linear-path flow targets and corresponding network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)

        t_data = self._expand_batch_scalar(t, x_0)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced linear flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced linear flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))


class GaussianFlowEDM(GaussianFlowAbstract):
    """Gaussian flow matching with EDM preconditioning."""
    _family = "flow"
    _loss_tag = "mse"

    def __init__(self, *args, sigma_data: float = 0.5, p_mean: float = -1.2, p_std: float = 1.2, **kwargs):
        super().__init__(*args, **kwargs)

        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive.")
        if p_std <= 0:
            raise ValueError("p_std must be positive.")

        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        sigma_min = float(max(self._eps, torch.finfo(self._fdtype).tiny))
        self._sample_timesteps = None
        if self._sampler != "adaptive_heun":
            self._sample_timesteps = build_flow_timesteps(
                self._schedule, self._sample_steps, self._sigma_max, sigma_min, self._device, self._fdtype,
                **self._schedule_kwargs,
            )

    def _check_sigma(self, x: torch.Tensor, sigma: torch.Tensor | float | None) -> torch.Tensor:
        """Return one positive noise level per batch element."""
        batch_size = x.shape[0]
        if sigma is None:
            return (self.p_mean + self.p_std * torch.randn(batch_size, device=x.device, dtype=x.dtype)).exp()

        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        elif sigma.shape == (batch_size, 1):
            sigma = sigma[:, 0]
        elif sigma.shape != (batch_size,):
            raise ValueError(f"sigma must be scalar or have shape ({batch_size},), got {tuple(sigma.shape)}.")
        if torch.any(sigma <= 0):
            raise ValueError("All sigma values must be positive.")
        return sigma

    def _coefficients(self, sigma: torch.Tensor, x: torch.Tensor):
        """Compute EDM preconditioning coefficients."""
        sigma_expanded = self._expand_batch_scalar(sigma, x)
        denominator = torch.sqrt(sigma_expanded.square() + self.sigma_data**2)
        c_noise = 0.25 * sigma.log().reshape(-1, 1)
        return (
            self.sigma_data**2 / denominator.square(),
            sigma_expanded * self.sigma_data / denominator,
            1.0 / denominator,
            c_noise,
            denominator,
            sigma_expanded,
        )

    def denoise(self, x_sigma: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Return the EDM clean-data prediction D_theta."""
        x_sigma = x_sigma.to(device=self._device, dtype=self._fdtype)
        sigma = self._check_sigma(x_sigma, sigma)
        c_skip, c_out, c_in, c_noise, _, _ = self._coefficients(sigma, x_sigma)
        residual = self._net(c_in * x_sigma, c_noise)

        if residual.shape != x_sigma.shape:
            raise ValueError(f"Network output has shape {tuple(residual.shape)}, expected {tuple(x_sigma.shape)}.")
        return c_skip * x_sigma + c_out * residual

    def vector_field(self, x_sigma: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Return dx/dsigma for the EDM probability-flow ODE."""
        x_sigma = x_sigma.to(device=self._device, dtype=self._fdtype)
        sigma = self._check_sigma(x_sigma, sigma)
        _, _, c_in, c_noise, denominator, sigma_expanded = self._coefficients(sigma, x_sigma)
        residual = self._net(c_in * x_sigma, c_noise)

        if residual.shape != x_sigma.shape:
            raise ValueError(f"Network output has shape {tuple(residual.shape)}, expected {tuple(x_sigma.shape)}.")
        # Equivalent to (x_sigma - denoise(x_sigma, sigma)) / sigma, but avoids
        # cancellation when sigma is small.
        return sigma_expanded / denominator.square() * x_sigma - self.sigma_data / denominator * residual

    def _precompute_loss(self, x: torch.Tensor, z: torch.Tensor | None, t: torch.Tensor | float | None = None):
        """Build EDM-preconditioned regression targets.

        The argument `t` is interpreted as sigma, not as a normalized time.
        """
        x = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x shape (N, *{self._sample_shape}), got {tuple(x.shape)}")

        z = torch.randn_like(x) if z is None else z.to(device=self._device, dtype=self._fdtype)
        if z.shape != x.shape:
            raise ValueError(f"z has shape {tuple(z.shape)}, expected {tuple(x.shape)}.")

        sigma = self._check_sigma(x, t)
        _, _, c_in, c_noise, denominator, sigma_expanded = self._coefficients(sigma, x)
        x_sigma = x + sigma_expanded * z
        residual_hat = self._net(c_in * x_sigma, c_noise)
        residual_target = sigma_expanded / (self.sigma_data * denominator) * x_sigma - denominator / self.sigma_data * z

        if residual_hat.shape != x_sigma.shape:
            raise ValueError(f"Network output has shape {tuple(residual_hat.shape)}, expected {tuple(x_sigma.shape)}.")

        if residual_target.shape != residual_hat.shape:
            raise ValueError(f"Target has shape {tuple(residual_target.shape)}, but prediction has shape {tuple(residual_hat.shape)}.")

        return residual_hat, residual_target, sigma

    def _loss(self, x: torch.Tensor, z: torch.Tensor | None = None, t: torch.Tensor | float | None = None) -> torch.Tensor:
        """Return unreduced EDM losses."""
        prediction, target, sigma = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(prediction, target, sigma)

    def loss(self, x: torch.Tensor, z: torch.Tensor | None = None, t: torch.Tensor | float | None = None) -> torch.Tensor:
        """Compute the reduced EDM objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))

    @torch.inference_mode()
    def _sample(
        self,
        n_samples: int | None = None,
        return_trajectory: bool = True,
        sample_source: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Generate samples by integrating the EDM probability-flow ODE."""
        self._net.eval()
        _, x = self._resolve_sample_source(n_samples, sample_source)
        sigma_min = float(max(self._eps, torch.finfo(self._fdtype).tiny))
        if self._sampler == "adaptive_heun":
            if chunk_size is not None:
                raise ValueError("chunk_size is only supported for fixed-step flow samplers.")
            return sample_flow_adaptive_heun(
                self.vector_field, x, t_min=self._sigma_max, t_max=sigma_min, atol=self._atol, rtol=self._rtol,
                h_init=self._h_init, h_min=self._h_min, h_max=self._h_max, return_trajectory=return_trajectory,
            )
        return sample_flow(self.vector_field, x, self._sample_timesteps, sampler=self._sampler,
                           return_trajectory=return_trajectory, chunk_size=chunk_size)


class GaussianFlowOTLinear(GaussianFlowAbstract):
    """Gaussian source flow with a minibatch OT coupling and a linear path."""
    _family = "flow"
    _loss_tag = "mse"

    def __init__(self, *args, ot_method="exact", ot_reg=0.05, **kwargs):
        super().__init__(*args, **kwargs)

        if ot_method not in {"exact", "sinkhorn"}:
            raise ValueError(f"Unknown OT method: {ot_method}.")
        if float(ot_reg) <= 0.0:
            raise ValueError(f"ot_reg must be positive, got {ot_reg}.")

        self.ot_method = ot_method
        self.ot_reg = float(ot_reg)

    @torch.no_grad()
    def _sample_ot_coupling(self, x_0, x_1):
        """Sample pairs from the empirical minibatch OT plan."""
        n_0, n_1 = x_0.shape[0], x_1.shape[0]
        x_0_flat = x_0.reshape(n_0, -1)
        x_1_flat = x_1.reshape(n_1, -1)

        cost = torch.cdist(x_0_flat.float(), x_1_flat.float()).square().detach().cpu().double().numpy()
        a, b = np.full(n_0, 1.0 / n_0), np.full(n_1, 1.0 / n_1)

        if self.ot_method == "exact":
            plan = ot.emd(a, b, cost)
        else:
            plan = ot.sinkhorn(a, b, cost, reg=self.ot_reg, method="sinkhorn_log")

        probabilities = torch.as_tensor(plan, device=x_0.device, dtype=torch.float64).flatten()
        indices = torch.multinomial(probabilities / probabilities.sum(), num_samples=n_0, replacement=True)
        return x_0[torch.div(indices, n_1, rounding_mode="floor")], x_1[indices.remainder(n_1)]

    def _precompute_loss(self, x, z, t=None):
        """Build OT-coupled linear-path flow targets and network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)
        x_0, x_1 = self._sample_ot_coupling(x_0, x_1)

        t_data = self._expand_batch_scalar(t, x_0)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced OT flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced OT flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))


class GaussianFlowDDPM(GaussianFlowAbstract):
    """Gaussian-source flow on the VP diffusion path (beta schedule)."""
    _family = "flow"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.05,
        t_max: float = 0.95,
        cosine_s: float = 0.008,
        base_or_sample: torch.Tensor = None,
        sigma_max: float = 1.0,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = "cpu",
        sampler: str = "heun",
        schedule: str = "linear",
        sample_steps: int | None = None,
        image_seq_len: int | None = None,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        h_init: float | None = None,
        h_min: float = 1e-4,
        h_max: float = 0.1,
    ):
        """Initialize the VP-inspired Gaussian flow and its cosine schedule."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, sigma_max=sigma_max, fdtype=fdtype,
                         idtype=idtype, device=device, sampler=sampler, schedule=schedule,
                         sample_steps=sample_steps, image_seq_len=image_seq_len, base_shift=base_shift,
                         max_shift=max_shift, base_image_seq_len=base_image_seq_len,
                         max_image_seq_len=max_image_seq_len, atol=atol, rtol=rtol,
                         h_init=h_init, h_min=h_min, h_max=h_max)

        self._s0 = float(cosine_s)

    def _vp_coefs(self, t: torch.Tensor):
        """Evaluate continuous VP coefficients at the requested flow times."""
        s = (1.0 - t).clamp(0.0, 1.0)

        f = ((s + self._s0) / (1.0 + self._s0)) * (np.pi / 2.0)
        f0 = (self._s0 / (1.0 + self._s0)) * (np.pi / 2.0)

        abar = torch.cos(f).pow(2) / (np.cos(f0) ** 2)
        abar = abar.clamp(min=self._eps, max=1.0)

        beta_s = (np.pi / (1.0 + self._s0)) * torch.tan(f)
        beta_s = beta_s.clamp_min(0.0)

        a = torch.sqrt(abar.clamp_min(self._eps))
        sigma = torch.sqrt((1.0 - abar).clamp_min(self._eps))

        return beta_s, abar, a, sigma

    def _precompute_loss(self, x, z, t=None):
        """Build VP-path velocity targets and network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)

        beta_s, abar, a, sigma = (self._expand_batch_scalar(value, x_0) for value in self._vp_coefs(t))
        x_t = a * x_1 + sigma * x_0
        v_t = -0.5 * beta_s * (abar * x_t - a * x_1) / (1.0 - abar).clamp_min(self._eps)
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced VP flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced VP flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))
