"""Base classes for diffusion and flow-matching models."""

import math
import torch
from ._schedules import build_flow_timesteps, cosine_schedule
from ._solvers import sample_ddim, sample_ddpm, sample_flow, sample_flow_adaptive_heun


class Base:
    """Common base class for native generative models.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.float32,
        device: torch.device = 'cpu',
    ):
        """Initialize the shared model state and move the network to the target device."""
        if isinstance(dim, int):
            sample_shape = (int(dim),)
        else:
            sample_shape = tuple(int(axis) for axis in dim)
        if not sample_shape or any(axis <= 0 for axis in sample_shape):
            raise ValueError(f"dim must define a non-empty positive sample shape, got {dim}.")

        self._sample_shape = sample_shape
        self._dim = sample_shape[0] if len(sample_shape) == 1 else int(math.prod(sample_shape))
        self._base_or_sample = base_or_sample
        self._n_steps = int(n_steps)

        self._fdtype = fdtype
        self._idtype = idtype
        self._eps = torch.finfo(self._fdtype).eps
        self._device = torch.device(device)
        self._net = net.to(device=self._device, dtype=self._fdtype)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw default source samples for the model family."""
        raise NotImplementedError("'_sample_source_default' not implemented.")

    def _loss_fn(self, x_hat: torch.Tensor, x: torch.Tensor, t: int) -> torch.Tensor:
        """Compute unreduced per-sample losses at timestep `t`."""
        raise NotImplementedError("'_loss_fn' not implemented.")

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Reduce a batch of per-sample losses to a scalar objective."""
        raise NotImplementedError("'_reduce' not implemented.")

    def _sample_source(self, n_samples: int) -> torch.Tensor:
        """Sample source points from a fixed base tensor or the default source law."""

        if isinstance(self._base_or_sample, torch.Tensor):
            base = self._base_or_sample.to(device=self._device, dtype=self._fdtype)

            if tuple(base.shape) == self._sample_shape:
                samples = base.unsqueeze(0).expand(n_samples, *([-1] * len(self._sample_shape)))
            else:
                if tuple(base.shape[1:]) != self._sample_shape:
                    raise ValueError(f"base_or_sample has sample dim/shape {tuple(base.shape[1:])}, expected {self._sample_shape}")
                idx = torch.randint(0, base.size(0), (n_samples,), device=self._device)
                samples = base.index_select(0, idx)

        else:
            samples = self._sample_source_default(n_samples)
        return samples

    def _resolve_sample_source(
        self,
        n_samples: int | None,
        sample_source: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        """Return the initial samples for an inference sampler."""
        if sample_source is None:
            if n_samples is None:
                raise ValueError("n_samples is required when sample_source is not provided.")
            n_samples = int(n_samples)
            return n_samples, self._sample_source(n_samples)

        x = sample_source.to(device=self._device, dtype=self._fdtype)
        if x.ndim < 1 or tuple(x.shape[1:]) != self._sample_shape:
            raise ValueError(f"sample_source must have shape (N, *{self._sample_shape}), got {tuple(x.shape)}")

        source_n = int(x.shape[0])
        if n_samples is None:
            return source_n, x

        n_samples = int(n_samples)
        if source_n != n_samples:
            raise ValueError(f"sample_source has {source_n} samples but n_samples={n_samples}.")
        return n_samples, x

    def _expand_batch_scalar(self, value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reshape one scalar per batch item so it broadcasts over sample dimensions."""
        return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))

    def _check_t(self, t, n_samples: int) -> torch.Tensor:
        """Validate and normalize user-provided diffusion timesteps."""
        if isinstance(t, bool):
            raise TypeError("'t' must be an int, float, or tensor, not bool.")

        if isinstance(t, int):
            if not (1 <= t <= self._n_steps):
                raise ValueError(f"Integer 't' must be in [1, {self._n_steps}], got {t}.")
            return torch.full((n_samples,), t, device=self._device, dtype=self._idtype)

        if isinstance(t, float):
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"Float 't' must be in [0, 1], got {t}.")
            t_step = min(max(int(t * self._n_steps), 1), self._n_steps)
            return torch.full((n_samples,), t_step, device=self._device, dtype=self._idtype)

        if isinstance(t, torch.Tensor):
            t = t.to(device=self._device)
            if t.ndim == 0:
                return self._check_t(t.item(), n_samples)
            if t.ndim != 1 or t.numel() != n_samples:
                raise ValueError(f"Tensor 't' must have shape ({n_samples},), got {tuple(t.shape)}.")

            if torch.is_floating_point(t):
                if ((t < 0.0) | (t > 1.0)).any():
                    raise ValueError("Floating tensor 't' must have values in [0, 1].")
                t = (t * self._n_steps).to(dtype=self._idtype)
                return t.clamp_(1, self._n_steps)

            if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                if ((t < 1) | (t > self._n_steps)).any():
                    raise ValueError(f"Integer tensor 't' must have values in [1, {self._n_steps}].")
                return t.to(dtype=self._idtype)

            raise TypeError(f"Unsupported tensor dtype for 't': {t.dtype}.")

        raise TypeError(f"Unsupported type for 't': {type(t).__name__}.")


class DDPMAbstract(Base):
    """DDPM abstract."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        base_or_sample: torch.Tensor = None,
        sigma_max: float = 1.0,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = 'cpu',
        sampler: str = "ddpm",
        sample_steps: int | None = None,
        eta: float = 0.0,
    ):
        """Build the shared DDPM coefficients and posterior schedule."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)

        if float(sigma_max) <= 0.0:
            raise ValueError(f"sigma_max must be positive, got {sigma_max}.")
        self._sigma_max = float(sigma_max)
        self._sampler = sampler
        if self._sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"Unknown DDPM sampler {self._sampler!r}; expected 'ddpm' or 'ddim'.")
        self._sample_steps = None if sample_steps is None else int(sample_steps)
        if self._sample_steps is not None and not (1 <= self._sample_steps <= self._n_steps):
            raise ValueError(f"sample_steps must be in [1, {self._n_steps}], got {sample_steps}.")
        self._eta = float(eta)
        if self._eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {eta}.")

        self._alpha_bar, self._alphas, self._betas, self._sqrt_post_var = cosine_schedule(n_steps,
                                                                                          device,
                                                                                          fdtype,
                                                                                          idtype
                                                                                          )

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw Gaussian noise used as the DDPM source distribution."""
        return self._sigma_max * torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)

    def _latent(self, x_1: torch.Tensor, eps: torch.Tensor = None, t: int = None):
        """Construct noisy latent states and normalized times for DDPM training."""
        x_1 = x_1.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x1 shape (N, *{self._sample_shape}), got {tuple(x_1.shape)}")

        n_samples = x_1.size(0)
        if t is None:
            t = torch.randint(1, self._n_steps + 1, (n_samples,), device=self._device, dtype=self._idtype)
        else:  # if t is given
            t = self._check_t(t, n_samples)
        t_idx = (t - 1).to(device=self._device, dtype=self._idtype)
        t_norm = (t / self._n_steps).unsqueeze(-1)

        a_bar_t = self._alpha_bar.index_select(0, t_idx).unsqueeze(-1)

        eps = self._sample_source(n_samples) if eps is None else eps.to(device=self._device,
                                                                        dtype=self._fdtype)
        if eps.shape != x_1.shape:
            raise ValueError(f"eps must have shape {tuple(x_1.shape)}, got {tuple(eps.shape)}")

        a_bar_t = self._expand_batch_scalar(a_bar_t, x_1)
        x_t = torch.sqrt(a_bar_t) * x_1 + torch.sqrt(1.0 - a_bar_t) * eps

        return x_1, x_t, eps, t_norm, t_idx, a_bar_t

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Average DDPM losses over the batch."""
        return loss_values.mean()

    @torch.inference_mode()
    def _sample(
        self,
        n_samples: int | None = None,
        return_trajectory: bool = True,
        sample_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Dispatch to a native DDPM inference sampler."""
        if self._sampler == "ddpm":
            return sample_ddpm(self, n_samples, return_trajectory=return_trajectory, sample_source=sample_source)
        return sample_ddim(
            self,
            n_samples,
            n_steps=self._sample_steps,
            eta=self._eta,
            return_trajectory=return_trajectory,
            sample_source=sample_source,
        )

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int | None = None,
        sample_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples by running the reverse DDPM process."""
        x, _ = self._sample(n_samples, return_trajectory=False, sample_source=sample_source)
        return x


class FlowAbstract(Base):
    """Abstract Flow Matching."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.01,
        t_max: float = 0.99,
        base_or_sample: torch.Tensor = None,
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
        """Initialize a continuous-time flow model on a bounded time interval."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)

        self._t_min = float(t_min)
        self._t_max = float(t_max)
        if not (0.0 <= self._t_min < self._t_max <= 1.0):
            raise ValueError(f"Need 0 <= t_min < t_max <= 1, got {self._t_min}, {self._t_max}")

        self._sampler = sampler
        if self._sampler not in {"euler", "heun", "rk4", "adaptive_heun"}:
            raise ValueError(f"Unknown flow sampler {self._sampler!r}.")

        self._schedule = schedule
        if self._schedule not in {"linear", "quadratic", "flux_shifted"}:
            raise ValueError(f"Unknown flow timestep schedule {self._schedule!r}.")

        self._sample_steps = self._n_steps if sample_steps is None else int(sample_steps)
        if self._sample_steps < 1:
            raise ValueError(f"sample_steps must be at least 1, got {sample_steps}.")
        self._schedule_kwargs = {}
        if self._schedule == "flux_shifted":
            if image_seq_len is None:
                raise ValueError("image_seq_len is required for schedule='flux_shifted'.")
            self._schedule_kwargs = {
                "image_seq_len": int(image_seq_len),
                "base_shift": float(base_shift),
                "max_shift": float(max_shift),
                "base_image_seq_len": int(base_image_seq_len),
                "max_image_seq_len": int(max_image_seq_len),
            }

        self._atol = float(atol)
        self._rtol = float(rtol)
        self._h_init = h_init
        self._h_min = float(h_min)
        self._h_max = float(h_max)
        self._sample_timesteps = None
        if self._sampler != "adaptive_heun":
            self._sample_timesteps = build_flow_timesteps(
                schedule=self._schedule,
                n_steps=self._sample_steps,
                t_min=self._t_min,
                t_max=self._t_max,
                device=self._device,
                dtype=self._fdtype,
                **self._schedule_kwargs,
            )

    def _t(self, n_samples):
        """Draw random continuous times in the configured training interval."""
        t = torch.rand((n_samples, 1), device=self._device, dtype=self._fdtype)
        return self._t_min + (self._t_max - self._t_min) * t

    def _check_t(self, t, n_samples: int) -> torch.Tensor:
        """Validate and normalize user-provided flow times."""
        if t is None:
            return self._t(n_samples)

        if isinstance(t, bool):
            raise TypeError("'t' must be an int, float, or tensor, not bool.")

        if isinstance(t, int):
            if not (0 <= t <= self._n_steps):
                raise ValueError(f"Integer 't' must be in [0, {self._n_steps}], got {t}.")
            t_value = self._t_min + (self._t_max - self._t_min) * (t / max(self._n_steps, 1))
            return torch.full((n_samples, 1), t_value, device=self._device, dtype=self._fdtype)

        if isinstance(t, float):
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"Float 't' must be in [0, 1], got {t}.")
            return torch.full((n_samples, 1), t, device=self._device, dtype=self._fdtype)

        if isinstance(t, torch.Tensor):
            t = t.to(device=self._device)
            if t.ndim == 0:
                return self._check_t(t.item(), n_samples)
            if t.ndim == 1:
                if t.numel() != n_samples:
                    raise ValueError(f"Tensor 't' must have shape ({n_samples},) or ({n_samples}, 1), got {tuple(t.shape)}.")
                t = t.unsqueeze(-1)
            elif t.ndim != 2 or t.shape != (n_samples, 1):
                raise ValueError(f"Tensor 't' must have shape ({n_samples},) or ({n_samples}, 1), got {tuple(t.shape)}.")

            if torch.is_floating_point(t):
                if ((t < 0.0) | (t > 1.0)).any():
                    raise ValueError("Floating tensor 't' must have values in [0, 1].")
                return t.to(dtype=self._fdtype)

            if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                if ((t < 0) | (t > self._n_steps)).any():
                    raise ValueError(f"Integer tensor 't' must have values in [0, {self._n_steps}].")
                t = t.to(dtype=self._fdtype)
                return self._t_min + (self._t_max - self._t_min) * (t / max(self._n_steps, 1))

            raise TypeError(f"Unsupported tensor dtype for 't': {t.dtype}.")

        raise TypeError(f"Unsupported type for 't': {type(t).__name__}.")

    def _latent(self, x_1: torch.Tensor, x_0: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Prepare paired source/target samples and a time batch for flow losses."""
        x_1 = x_1.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x1 shape (N, *{self._sample_shape}), got {tuple(x_1.shape)}")

        n_samples = x_1.size(0)
        if t is None:  # default to uniform sampling
            t = self._t(n_samples)
        else:  # if t is given
            t = self._check_t(t, n_samples)

        x_0 = self._sample_source(n_samples) if x_0 is None else x_0.to(device=self._device,
                                                                        dtype=self._fdtype)
        if x_0.shape != x_1.shape:
            raise ValueError(f"x0 must have shape {tuple(x_1.shape)}, got {tuple(x_0.shape)}")

        return x_0, x_1, t

    @torch.inference_mode()
    def _sample(
        self,
        n_samples: int | None = None,
        return_trajectory: bool = True,
        sample_source: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Dispatch to a flow inference schedule and ODE solver."""
        self._net.eval()
        n_samples, x = self._resolve_sample_source(n_samples, sample_source)
        if self._sampler == "adaptive_heun":
            if chunk_size is not None:
                raise ValueError("chunk_size is only supported for fixed-step flow samplers.")
            return sample_flow_adaptive_heun(
                self._net,
                x,
                t_min=self._t_min,
                t_max=self._t_max,
                atol=self._atol,
                rtol=self._rtol,
                h_init=self._h_init,
                h_min=self._h_min,
                h_max=self._h_max,
                return_trajectory=return_trajectory,
            )

        return sample_flow(self._net, x, self._sample_timesteps, sampler=self._sampler,
                           return_trajectory=return_trajectory, chunk_size=chunk_size)

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int | None = None,
        sample_source: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Generate samples by integrating the learned flow field."""
        x, _ = self._sample(n_samples, return_trajectory=False, sample_source=sample_source, chunk_size=chunk_size)
        return x


class GaussianFlowAbstract(FlowAbstract):
    """Abstract Gaussian Flow Matching."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.01,
        t_max: float = 0.99,
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
        """Initialize a Gaussian-source flow with a configurable source scale."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, fdtype=fdtype, idtype=idtype,
                         device=device, sampler=sampler, schedule=schedule, sample_steps=sample_steps,
                         image_seq_len=image_seq_len, base_shift=base_shift, max_shift=max_shift,
                         base_image_seq_len=base_image_seq_len, max_image_seq_len=max_image_seq_len,
                         atol=atol, rtol=rtol, h_init=h_init, h_min=h_min, h_max=h_max)

        if float(sigma_max) <= 0.0:
            raise ValueError(f"sigma_max must be positive, got {sigma_max}.")
        self._sigma_max = float(sigma_max)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw Gaussian source samples for Gaussian flow models."""
        return self._sigma_max * torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)

    def _loss_fn(self, u_t: torch.Tensor, v_t: torch.Tensor, t: int) -> torch.Tensor:
        """Compute pointwise mean-squared velocity errors."""
        return torch.nn.functional.mse_loss(u_t, v_t, reduction='none')

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Average Gaussian flow losses over all batch elements."""
        return loss_values.mean()
