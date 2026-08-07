"""Diffusion and DLPM model definitions."""

import warnings
import torch
from ._abs import Base, DDPMAbstract
from ._noise import sample_scaled_scalar_alpha_stable
from ._schedules import cosine_schedule


class DDPMEps(DDPMAbstract):
    """DDPM with eps-prediction parameterization."""
    _family = "diffusion"
    _loss_tag = "mse"

    def _loss_fn(self, eps_hat: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        """Return unreduced epsilon-prediction losses for a DDPM batch."""
        return torch.nn.functional.mse_loss(eps_hat, eps, reduction="none")

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Evaluate epsilon-prediction losses on noisy DDPM latents."""
        _, x_t, eps, t_norm, t_idx, _ = self._latent(x_1=x, eps=z, t=t)

        eps_hat = self._net(x_t, t_norm)
        if eps.shape != eps_hat.shape:
            raise ValueError(
                f"Shape mismatch: eps has shape {tuple(eps.shape)} but eps_hat has shape {tuple(eps_hat.shape)}."
            )

        return self._loss_fn(eps_hat, eps, t_idx)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced DDPM epsilon-prediction training loss."""
        return self._reduce(self._loss(x=x, z=z, t=t))

    def _get_eps_hat(self, x: torch.Tensor, t_norm: torch.Tensor, t_idx: int) -> torch.Tensor:
        """Return direct epsilon predictions for sampling."""
        return self._net(x, t_norm)


class DDPMV(DDPMAbstract):
    """DDPM with v-prediction parameterization."""
    _family = "diffusion"
    _loss_tag = "mse"

    def _loss_fn(self, v_hat: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """Return unreduced v-prediction losses for a DDPM batch."""
        return torch.nn.functional.mse_loss(v_hat, v, reduction="none")

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Build the DDPM v-target and evaluate the network on noisy inputs."""
        x_1, x_t, eps, t_norm, t_idx, a_bar_t = self._latent(x_1=x, eps=z, t=t)

        v = torch.sqrt(a_bar_t) * eps - torch.sqrt(1.0 - a_bar_t) * x_1

        v_hat = self._net(x_t, t_norm)
        if v.shape != v_hat.shape:
            raise ValueError(
                f"Shape mismatch: v has shape {tuple(v.shape)} but v_hat has shape {tuple(v_hat.shape)}."
            )

        return self._loss_fn(v_hat, v, t_idx)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced DDPM v-prediction training loss."""
        return self._reduce(self._loss(x=x, z=z, t=t))

    def _get_eps_hat(self, x: torch.Tensor, t_norm: torch.Tensor, t_idx: int) -> torch.Tensor:
        """Recover epsilon predictions from the learned v-parameterization."""
        return torch.sqrt(1.0 - self._alpha_bar[t_idx]) * x + torch.sqrt(self._alpha_bar[t_idx]) * self._net(x, t_norm)


class DDPMX0(DDPMAbstract):
    """DDPM with x0-prediction parameterization."""
    _family = "diffusion"
    _loss_tag = "mse"

    def _loss_fn(self, x0_hat: torch.Tensor, x0: torch.Tensor, t: int) -> torch.Tensor:
        """Return weighted x0-prediction losses for a DDPM batch."""
        alpha_bar = self._alpha_bar.index_select(0, t)
        w = (alpha_bar / (1.0 - alpha_bar)).view(-1, 1).clamp(min=1e-3, max=1e3)
        return w * (x0_hat - x0).square()

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Evaluate x0-prediction losses on noisy DDPM latents."""
        x_1, x_t, _, t_norm, t, _ = self._latent(x_1=x, eps=z, t=t)

        x_1_hat = self._net(x_t, t_norm)
        if x_1_hat.shape != x_1.shape:
            raise ValueError(f"Shape mismatch: x_1_hat={tuple(x_1_hat.shape)} vs"
                             f" x_1={tuple(x_1.shape)}")

        return self._loss_fn(x_1_hat, x_1, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced DDPM x0-prediction training loss."""
        return self._reduce(self._loss(x=x, z=z, t=t))

    def _get_eps_hat(self, x, t_norm, t_idx):
        """Convert x0 predictions back to epsilon predictions for sampling."""
        return (x - torch.sqrt(self._alpha_bar[t_idx]) * self._net(x, t_norm)) / torch.sqrt(1.0 - self._alpha_bar[t_idx])


class DLPMEps(Base):
    """DLPM with eps-prediction parameterization."""
    _family = "diffusion"
    _loss_tag = "sqrt_mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        alpha: float = 1.8,
        n_trial_A: int = 1,
        n_trial_G: int = 1,
        reduce_type: str = "mean",
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = 'cpu',
        sampler: str = "native",
    ):
        """Initialize the native DLPM epsilon model and its stable-noise schedule."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)
        self._sampler = sampler
        if self._sampler != "native":
            raise ValueError(f"DLPMEps only supports sampler='native', got {self._sampler!r}.")

        self._a = float(alpha)
        if not (0.0 < self._a <= 2.0):
            raise ValueError(f"'alpha' must be in (0,2], got {self._a}.")

        _, _, self._betas, _ = cosine_schedule(n_steps, device=device, fdtype=fdtype, idtype=idtype)
        self._gamma_t = (1.0 - self._betas).clamp_min(self._eps).pow(1.0 / self._a)
        self._sigma_t = (1.0 - self._gamma_t.pow(self._a)).clamp_min(0.0).pow(1.0 / self._a)
        self._gamma_1_t = torch.ones(self._n_steps + 1, device=self._device, dtype=self._fdtype)
        self._gamma_1_t[1:] = torch.cumprod(self._gamma_t, dim=0)
        self._sigma_1_t = (1.0 - self._gamma_1_t.pow(self._a)).clamp_min(0.0).pow(1.0 / self._a)

        self._n_trial_A = int(n_trial_A)
        self._n_trial_G = int(n_trial_G)

        if reduce_type not in ("mean", "median"):
            raise ValueError(f"'reduce_type' must be in ('mean','median'), got {reduce_type}.")
        self._reduce_type = reduce_type

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Aggregate Monte Carlo DLPM losses across stable and Gaussian draws."""
        loss_values = loss_values.reshape(self._n_trial_A, self._n_trial_G, self._n)
        if self._reduce_type == "median":
            loss_values = loss_values.mean(dim=1)  # mean over G trials
            loss_values, _ = loss_values.median(dim=0)  # median over A trials
        return loss_values.mean()  # mean over samples

    def _loss_fn(self, eps_hat: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        """Compute per-sample epsilon reconstruction errors."""
        # NOTE see L666 in gendynamics/_vendor/DLPM/dlpm/methods/GenerativeLevyProcess.py
        # NOTE see L272 in gendynamics/_vendor/DLPM/dlpm/methods/dlpm.py
        # NOTE We follow the released code rather than the paper here.
        loss_values = torch.nn.functional.mse_loss(eps_hat, eps, reduction="none")
        loss_values = loss_values.mean(dim=tuple(range(1, loss_values.ndim)))
        return loss_values.clamp_min(self._eps).sqrt()

    def _Sigma_1_t(self, A: torch.Tensor) -> torch.Tensor:
        """Build the vendor-aligned sampling-time Sigma path from one stable chain."""
        # NOTE see L307 in gendynamics/_vendor/DLPM/dlpm/methods/GenerativeLevyProcess.py
        S = torch.zeros((self._n_steps, A.shape[1]), device=A.device, dtype=A.dtype)
        for t in range(1, self._n_steps):
            S[t] = self._sigma_t[t - 1].square() * A[t]
            S[t] += self._gamma_t[t - 1].square() * S[t - 1]
        return S

    def _g_Sigma_hat_Gamma(self, Sigma_1_t: torch.Tensor, t: int):
        """Compute vendor-aligned reverse-step coefficients from the Sigma path."""
        # NOTE see L272 in gendynamics/_vendor/DLPM/dlpm/methods/dlpm.py
        Sigma_ratio = Sigma_1_t[t - 1] / Sigma_1_t[t].clamp_min(self._eps)
        Gamma_t = 1.0 - Sigma_ratio * self._gamma_t[t - 1].square()
        Gamma_t = Gamma_t.clamp(0.0, 1.0)
        Sigma_hat = (Gamma_t * Sigma_1_t[t - 1]).clamp_min(0.0)
        return Sigma_hat, self._gamma_t[t - 1], Gamma_t

    def _expand(self, x):  # since we stack the Monte Carlo drawing in the different dimension
        """Broadcast a batch across the configured Monte Carlo axes."""
        return x.expand(self._n_trial_A, self._n_trial_G, self._n)

    def _draw_A(self, n):
        """Sample positive stable mixing coefficients for DLPM noise."""
        return sample_scaled_scalar_alpha_stable(n_samples=n, alpha=self._a, device=self._device, dtype=self._fdtype)

    def _sample_source_default(
        self,
        n_samples: int,
        *,
        expand_trials: bool = False,
        return_A: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Draw alpha-stable source samples, optionally expanded for Monte Carlo loss."""
        if expand_trials:
            A = self._draw_A(self._n_trial_A * n_samples)
            A = A.view(self._n_trial_A, 1, n_samples).expand(self._n_trial_A, self._n_trial_G, n_samples).reshape(-1)
            G = torch.randn(
                self._n_trial_A * self._n_trial_G * n_samples,
                *self._sample_shape,
                device=self._device,
                dtype=self._fdtype,
            )
            scale = self._expand_batch_scalar(A, G).sqrt()
            if return_A:
                return scale * G, A
            return scale * G

        A = self._draw_A(n_samples).reshape(n_samples)
        G = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
        return self._expand_batch_scalar(A, G).sqrt() * G

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Evaluate unreduced DLPM epsilon losses with internal Monte Carlo sampling."""
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x shape (N, *{self._sample_shape}), got {tuple(x.shape)}")
        self._n = x_1.size(0)

        if z is not None:
            warnings.warn("In 'DLPMEps.loss', input 'z' is ignored (noise is sampled internally).")

        if t is None:  # default to uniform sampling
            t = torch.randint(1, self._n_steps, (self._n,), device=self._device, dtype=self._idtype)
        else:  # if t is given
            t = self._check_t(t, self._n)

        t_e = self._expand(t.view(1, 1, self._n)).reshape(-1)                                       # (_n_trial_A * _n_trial_G * n,)
        t_norm = self._expand((t / self._n_steps).view(1, 1, self._n)).reshape(-1, 1)               # (_n_trial_A * _n_trial_G * n, 1)

        eps = self._sample_source_default(self._n, expand_trials=True)                              # (_n_trial_A * _n_trial_G * n, dim)

        x_1_e = x_1.view(1, 1, self._n, *self._sample_shape).expand(
            self._n_trial_A,
            self._n_trial_G,
            self._n,
            *self._sample_shape,
        )
        x_1_e = x_1_e.reshape(-1, *self._sample_shape)
        gamma_1_t = self._expand_batch_scalar(self._gamma_1_t.index_select(0, t_e), x_1_e)
        sigma_1_t = self._expand_batch_scalar(self._sigma_1_t.index_select(0, t_e), x_1_e)

        x_t = gamma_1_t * x_1_e + sigma_1_t * eps

        eps_hat = self._net(x_t, t_norm)

        return self._loss_fn(eps_hat, eps, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced DLPM-eps training loss."""
        return self._reduce(self._loss(x=x, z=z, t=t))

    @torch.no_grad()
    def _sample(
        self,
        n_samples: int,
        return_trajectory: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run the native DLPM-eps reverse chain."""
        self._net.eval()

        # Sample latent stable path A_{1:T} used to build Sigma_{1->t}(A_{1:t})
        A_path = torch.stack([self._draw_A(n_samples).squeeze(-1) for _ in range(self._n_steps)], dim=0)
        Sigma_1_t = self._Sigma_1_t(A_path)

        eps = self._sample_source_default(n_samples)

        x = self._sigma_1_t[self._n_steps - 1] * eps  # vendor terminal barsigma
        l_x = [x] if return_trajectory else None

        # Reverse recursion (Table 4 DLPM): mean update divided by gamma_t, then add Gaussian innovation
        for t in range(self._n_steps - 1, 0, -1):  # t from T - 1 to 0

            t_norm = torch.full((n_samples, 1), t / self._n_steps, device=self._device, dtype=self._fdtype)
            Sigma_hat, gamma_t, Gamma_t = self._g_Sigma_hat_Gamma(Sigma_1_t, t)

            # NOTE see L276 in gendynamics/_vendor/DLPM/dlpm/methods/dlpm.py
            # NOTE We follow the released code rather than the paper here.
            eps_hat = self._net(x, t_norm)
            gamma_t_data = self._expand_batch_scalar(Gamma_t, x)
            x = (x - gamma_t_data * self._sigma_1_t[t] * eps_hat) / gamma_t

            if t > 1:
                innovation = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
                x = x + self._expand_batch_scalar(Sigma_hat.sqrt(), innovation) * innovation

            if return_trajectory:
                l_x.append(x)

        return x, l_x

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples with the native DLPM reverse sampler."""
        x, _ = self._sample(n_samples, return_trajectory=False)
        return x
