"""Shared test helpers."""

from pathlib import Path
import numpy as np
import torch


def _devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


class _AffineTimeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = torch.nn.Parameter(torch.tensor(0.35))
        self.wt = torch.nn.Parameter(torch.tensor(-0.2))
        self.b = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.wx * x + self.wt * t + self.b


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _reset_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_grads(loss: torch.Tensor, model: torch.nn.Module) -> list[torch.Tensor]:
    return [grad.detach().clone() for grad in torch.autograd.grad(loss, tuple(model.parameters()))]


def _assert_tensors_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def _assert_grad_lists_close(actual: list[torch.Tensor], expected: list[torch.Tensor], atol: float = 1e-6, rtol: float = 1e-6):
    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_tensors_close(actual_grad, expected_grad, atol=atol, rtol=rtol)


def _make_flow_matching_vendor(tmp_path: Path, *, exact: bool = False) -> Path:
    root = tmp_path / ("flow_matching_exact_vendor" if exact else "flow_matching_vendor")
    solver_body = """
class ODESolver:
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model

    def sample(self, x_init, step_size, method, time_grid, return_intermediates=False, **kwargs):
        x = x_init
        t = time_grid[:1].expand(x.size(0))
        for _ in range(max(int(round((time_grid[-1] - time_grid[0]).item() / step_size)), 0)):
            v0 = self.velocity_model(x=x, t=t)
            t_next = (t + step_size).clamp_max(time_grid[-1])
            x_euler = x + step_size * v0
            v1 = self.velocity_model(x=x_euler, t=t_next)
            x = x + 0.5 * step_size * (v0 + v1)
            t = t_next
        return x
""" if exact else """
class ODESolver:
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model

    def sample(self, x_init, step_size, method, time_grid, return_intermediates=False, **kwargs):
        x = x_init
        t = time_grid[-1].expand(x.size(0))
        return x + (time_grid[-1] - time_grid[0]) * self.velocity_model(x=x, t=t)
"""
    _write(root / "flow_matching" / "__init__.py", "")
    _write(
        root / "flow_matching" / "path" / "__init__.py",
        """
import torch
from types import SimpleNamespace


class AffineProbPath:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, x_0, x_1, t):
        x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        dx_t = x_1 - x_0
        return SimpleNamespace(x_t=x_t, dx_t=dx_t, t=t, x_0=x_0, x_1=x_1)
""",
    )
    _write(root / "flow_matching" / "path" / "scheduler.py", "class CondOTScheduler:\n    pass\n")
    _write(root / "flow_matching" / "solver" / "__init__.py", solver_body)
    _write(
        root / "flow_matching" / "utils" / "__init__.py",
        """
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
""",
    )
    return root


def _make_score_sde_vendor(tmp_path: Path, mode: str | None = None) -> Path:
    root = tmp_path / ("score_sde_vendor" if mode is None else f"score_sde_vendor_{mode}")
    sde_body = f"""
MODE = {mode!r}


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=100):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.N = int(N)
        self.mode = MODE
""" if mode is not None else """
import torch


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=100):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.N = int(N)

    def marginal_prob(self, x, t):
        std = torch.full((x.size(0),), self.sigma_min, device=x.device, dtype=x.dtype)
        return x, std

    def prior_sampling(self, shape):
        return self.sigma_max * torch.randn(*shape)

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        diffusion = torch.full((x.size(0),), self.sigma_max, device=x.device, dtype=x.dtype)
        return drift, diffusion
"""
    losses_body = """
import torch
from gendynamics._schedules import cosine_schedule


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=False, eps=1e-3):
    def loss_fn(model, batch):
        x_1 = batch.view(batch.size(0), -1)
        t = torch.randint(1, sde.N + 1, (x_1.size(0),), device=x_1.device, dtype=torch.int64)
        alpha_bar, _, _, _ = cosine_schedule(sde.N, x_1.device, x_1.dtype, torch.int64)
        a_bar_t = alpha_bar.index_select(0, t - 1).unsqueeze(-1)
        noise = torch.randn_like(x_1)
        x_t = torch.sqrt(a_bar_t) * x_1 + torch.sqrt(1.0 - a_bar_t) * noise
        t_norm = (t.to(dtype=x_1.dtype) / sde.N).unsqueeze(-1)
        pred = model.model(x_t, t_norm)
        if sde.mode == "v":
            target = torch.sqrt(a_bar_t) * noise - torch.sqrt(1.0 - a_bar_t) * x_1
            return torch.nn.functional.mse_loss(pred, target, reduction="none").mean()
        weight = (a_bar_t / (1.0 - a_bar_t)).clamp(min=1e-3, max=1e3)
        return (weight * (pred - x_1).square()).mean()
    return loss_fn
""" if mode is not None else """
def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=False, eps=1e-3):
    def loss_fn(model, batch):
        out = model(batch, batch.new_zeros(batch.size(0)))
        return out.square().mean()
    return loss_fn
"""
    sampling_body = """
import torch
from gendynamics._schedules import cosine_schedule


class ReverseDiffusionPredictor:
    pass


class NoneCorrector:
    pass


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cpu'):
    def sampler(model):
        n_samples, dim = int(shape[0]), int(shape[1])
        x = torch.randn((n_samples, dim), device=device)
        alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(sde.N, x.device, x.dtype, torch.int64)
        for t in range(sde.N, 0, -1):
            t_idx = t - 1
            t_norm = torch.full((n_samples, 1), t / sde.N, device=x.device, dtype=x.dtype)
            raw = model.model(x, t_norm)
            if sde.mode == "v":
                eps_hat = torch.sqrt(1.0 - alpha_bar[t_idx]) * x + torch.sqrt(alpha_bar[t_idx]) * raw
            else:
                eps_hat = (x - torch.sqrt(alpha_bar[t_idx]) * raw) / torch.sqrt(1.0 - alpha_bar[t_idx])
            x = (x - betas[t_idx] / torch.sqrt(1.0 - alpha_bar[t_idx]) * eps_hat) / torch.sqrt(alphas[t_idx])
            if t > 1:
                x = x + sqrt_post_var[t_idx] * torch.randn_like(x)
        return inverse_scaler(x.view(n_samples, dim, 1, 1)), sde.N
    return sampler
""" if mode is not None else """
import torch


class ReverseDiffusionPredictor:
    pass


class NoneCorrector:
    pass


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cpu'):
    def sampler(model):
        x = torch.zeros(shape, device=device)
        return inverse_scaler(x), sde.N
    return sampler
"""
    _write(root / "sde_lib.py", sde_body)
    _write(root / "losses.py", losses_body)
    _write(root / "sampling.py", sampling_body)
    return root


def _make_dlpm_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "DLPM"
    _write(root / "dlpm" / "__init__.py", "")
    _write(root / "dlpm" / "methods" / "__init__.py", "")
    _write(
        root / "dlpm" / "methods" / "GenerativeLevyProcess.py",
        """
from types import SimpleNamespace
import torch


class _Params:
    def setParams(self, **kwargs):
        self.kwargs = dict(kwargs)


class GenerativeLevyProcess:
    def __init__(self, alpha, device, reverse_steps, time_spacing, rescale_timesteps, isotropic, scale):
        self.dlpm = SimpleNamespace(gen_a=_Params(), gen_eps=_Params())

    def training_losses(self, models, x_start, **kwargs):
        if model := models.get("default"):
            return {"loss": model(x_start, torch.ones(x_start.size(0), device=x_start.device, dtype=x_start.dtype)).square().mean()}
        return {"loss": x_start.square().mean()}

    def p_sample_loop(self, model, shape, progress=False):
        x = torch.randn(shape)
        for t in range(8, 0, -1):
            time = torch.full((shape[0],), t / 8.0, dtype=x.dtype)
            x = x - 0.1 * model(x, time)
        return x
""",
    )
    return root


def _make_tedm_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "physicsnemo_vendor"
    _write(root / "physicsnemo" / "__init__.py", "")
    _write(root / "physicsnemo" / "diffusion" / "__init__.py", "")
    _write(
        root / "physicsnemo" / "diffusion" / "noise_schedulers.py",
        """
import torch


class StudentTEDMNoiseScheduler:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, nu=10, sigma_data=0.5, P_mean=-1.2, P_std=1.2):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.nu = float(nu)
        self.sigma_data = float(sigma_data)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)

    def sample_time(self, batch_size, device=None, dtype=None):
        return torch.full((int(batch_size),), 0.5, device=device, dtype=dtype)

    def sigma(self, t):
        return torch.full_like(t, self.sigma_data)

    def add_noise(self, x, t):
        return x + self.sigma(t).reshape(t.numel(), *([1] * (x.ndim - 1))) * torch.ones_like(x)

    def loss_weight(self, t):
        return torch.ones_like(t)

    def timesteps(self, n_steps, device=None, dtype=None):
        return torch.linspace(1.0, 0.0, int(n_steps), device=device, dtype=dtype)

    def init_latents(self, shape, t, device=None, dtype=None):
        batch = int(t.numel())
        spatial_shape = tuple(shape)
        return self.sigma(t).reshape(batch, *([1] * len(spatial_shape))) * torch.ones((batch,) + spatial_shape, device=device, dtype=dtype)

    def get_denoiser(self, x0_predictor):
        def denoiser(x, sigma=None, **kwargs):
            if sigma is None:
                sigma = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return x0_predictor(x, sigma, **kwargs)
        return denoiser
""",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "preconditioners.py",
        """
import torch.nn as nn


class EDMPreconditioner(nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = float(sigma_data)

    def forward(self, x, t, condition=None, **kwargs):
        return self.model(x, t, condition=condition, **kwargs)
""",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "samplers.py",
        """
def sample(denoiser, x, scheduler, num_steps=18, solver="edm_stochastic_heun"):
    sigma = scheduler.timesteps(num_steps, device=x.device, dtype=x.dtype)[0].expand(x.size(0))
    return denoiser(x, sigma)
""",
    )
    return root


def _make_tedm_vendor_package_layout(tmp_path: Path) -> Path:
    root = tmp_path / "physicsnemo_vendor_pkg"
    _write(root / "physicsnemo" / "__init__.py", "")
    _write(root / "physicsnemo" / "diffusion" / "__init__.py", "")
    _write(
        root / "physicsnemo" / "diffusion" / "noise_schedulers" / "__init__.py",
        "raise RuntimeError('noise_schedulers package initializer should not run')\n",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "noise_schedulers" / "noise_schedulers.py",
        """
import torch


class StudentTEDMNoiseScheduler:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, nu=10, sigma_data=0.5, P_mean=-1.2, P_std=1.2):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.nu = float(nu)
        self.sigma_data = float(sigma_data)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)

    def sample_time(self, batch_size, device=None, dtype=None):
        return torch.full((int(batch_size),), 0.5, device=device, dtype=dtype)

    def sigma(self, t):
        return torch.full_like(t, self.sigma_data)

    def add_noise(self, x, t):
        return x + self.sigma(t).reshape(t.numel(), *([1] * (x.ndim - 1))) * torch.ones_like(x)

    def loss_weight(self, t):
        return torch.ones_like(t)

    def timesteps(self, n_steps, device=None, dtype=None):
        return torch.linspace(1.0, 0.0, int(n_steps), device=device, dtype=dtype)

    def init_latents(self, shape, t, device=None, dtype=None):
        batch = int(t.numel())
        spatial_shape = tuple(shape)
        return self.sigma(t).reshape(batch, *([1] * len(spatial_shape))) * torch.ones((batch,) + spatial_shape, device=device, dtype=dtype)

    def get_denoiser(self, x0_predictor):
        def denoiser(x, sigma=None, **kwargs):
            if sigma is None:
                sigma = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return x0_predictor(x, sigma, **kwargs)
        return denoiser
""",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "preconditioners" / "__init__.py",
        "raise RuntimeError('preconditioners package initializer should not run')\n",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "preconditioners" / "preconditioners.py",
        """
import torch.nn as nn


class EDMPreconditioner(nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = float(sigma_data)

    def forward(self, x, t, condition=None, **kwargs):
        return self.model(x, t, condition=condition, **kwargs)
""",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "samplers" / "__init__.py",
        "raise RuntimeError('samplers package initializer should not run')\n",
    )
    _write(
        root / "physicsnemo" / "diffusion" / "samplers" / "samplers.py",
        """
def sample(denoiser, x, scheduler, num_steps=18, solver="edm_stochastic_heun"):
    sigma = scheduler.timesteps(num_steps, device=x.device, dtype=x.dtype)[0].expand(x.size(0))
    return denoiser(x, sigma)
""",
    )
    return root
