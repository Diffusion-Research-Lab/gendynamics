"""Inference-time samplers for native gendynamics models."""

from typing import Any
import torch


def _time_batch(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        return torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
    return t.to(device=x.device, dtype=x.dtype).reshape(1, 1).expand(x.size(0), 1)


def _check_timesteps(timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("timesteps must be a 1D tensor with at least two entries.")
    return timesteps


def _check_chunk_size(chunk_size: int | None, n_samples: int) -> int | None:
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an int or None, not bool.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    return None if chunk_size >= n_samples else chunk_size


def _sample_flow_chunks(
    sample_chunk,
    x: torch.Tensor,
    chunk_size: int,
    return_trajectory: bool,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    x_out_chunks = []
    trajectory_chunks = None
    for x_chunk in x.split(chunk_size):
        x_out, trajectory = sample_chunk(x_chunk)
        x_out_chunks.append(x_out)
        if return_trajectory:
            if trajectory_chunks is None:
                trajectory_chunks = [[] for _ in trajectory]
            for chunks_at_time, value in zip(trajectory_chunks, trajectory):
                chunks_at_time.append(value)

    x_out = torch.cat(x_out_chunks, dim=0)
    trajectory_out = None if trajectory_chunks is None else [torch.cat(chunks, dim=0) for chunks in trajectory_chunks]
    return x_out, trajectory_out


@torch.inference_mode()
def sample_flow_euler(
    net: torch.nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    return_trajectory: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Integrate a flow ODE with explicit Euler steps."""
    timesteps = _check_timesteps(timesteps, x)
    trajectory = [x] if return_trajectory else None
    for i in range(timesteps.numel() - 1):
        t = _time_batch(timesteps[i], x)
        dt = timesteps[i + 1] - timesteps[i]
        x = x + dt * net(x, t)
        if trajectory is not None:
            trajectory.append(x)
    return x, trajectory


@torch.inference_mode()
def sample_flow_heun(
    net: torch.nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    return_trajectory: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Integrate a flow ODE with Heun's trapezoidal method."""
    timesteps = _check_timesteps(timesteps, x)
    trajectory = [x] if return_trajectory else None
    for i in range(timesteps.numel() - 1):
        t = _time_batch(timesteps[i], x)
        t_next = _time_batch(timesteps[i + 1], x)
        dt = timesteps[i + 1] - timesteps[i]
        v0 = net(x, t)
        x_euler = x + dt * v0
        v1 = net(x_euler, t_next)
        x = x + 0.5 * dt * (v0 + v1)
        if trajectory is not None:
            trajectory.append(x)
    return x, trajectory


@torch.inference_mode()
def sample_flow_rk4(
    net: torch.nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    return_trajectory: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Integrate a flow ODE with classical fourth-order Runge-Kutta steps."""
    timesteps = _check_timesteps(timesteps, x)
    trajectory = [x] if return_trajectory else None
    for i in range(timesteps.numel() - 1):
        t0 = timesteps[i]
        t1 = timesteps[i + 1]
        tm = 0.5 * (t0 + t1)
        dt = t1 - t0
        k1 = net(x, _time_batch(t0, x))
        k2 = net(x + 0.5 * dt * k1, _time_batch(tm, x))
        k3 = net(x + 0.5 * dt * k2, _time_batch(tm, x))
        k4 = net(x + dt * k3, _time_batch(t1, x))
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if trajectory is not None:
            trajectory.append(x)
    return x, trajectory


@torch.inference_mode()
def sample_flow_adaptive_heun(
    net: torch.nn.Module,
    x: torch.Tensor,
    t_min: float,
    t_max: float,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    h_init: float | None = None,
    h_min: float = 1e-4,
    h_max: float = 0.1,
    return_trajectory: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Integrate a flow ODE with adaptive Heun steps."""
    if atol <= 0.0 or rtol <= 0.0:
        raise ValueError("atol and rtol must be positive.")
    if h_min <= 0.0 or h_max <= 0.0 or h_min > h_max:
        raise ValueError("Need 0 < h_min <= h_max.")

    t = float(t_min)
    target = float(t_max)
    direction = 1.0 if target >= t else -1.0
    span = abs(target - t)
    h = min(h_max, max(h_min, span / 10.0)) if h_init is None else abs(float(h_init))
    h = min(h_max, max(h_min, h))
    trajectory = [x] if return_trajectory else None

    n_attempts = 0
    while direction * (target - t) > 0.0:
        n_attempts += 1
        if n_attempts > 1_000_000:
            raise RuntimeError("adaptive Heun solver exceeded 1,000,000 step attempts.")

        h_step = direction * min(h, abs(target - t))
        t_next = t + h_step
        v0 = net(x, _time_batch(t, x))
        x_euler = x + h_step * v0
        v1 = net(x_euler, _time_batch(t_next, x))
        x_heun = x + 0.5 * h_step * (v0 + v1)

        scale = atol + rtol * torch.maximum(x.abs(), x_heun.abs())
        err = ((x_heun - x_euler) / scale).abs().max().item()
        accept = err <= 1.0 or h <= h_min

        if accept:
            x = x_heun
            t = t_next
            if trajectory is not None:
                trajectory.append(x)

        factor = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * err ** -0.5))
        h = min(h_max, max(h_min, h * factor))

    return x, trajectory


@torch.inference_mode()
def sample_flow(
    net: torch.nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    sampler: str = "heun",
    return_trajectory: bool = True,
    chunk_size: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Dispatch to a named flow ODE sampler."""
    name = sampler
    chunk_size = _check_chunk_size(chunk_size, x.size(0))
    if chunk_size is not None:
        if name == "adaptive_heun":
            raise ValueError("chunk_size is only supported for fixed-step flow samplers.")
        return _sample_flow_chunks(
            lambda x_chunk: sample_flow(net, x_chunk, timesteps, sampler, return_trajectory, chunk_size=None, **kwargs),
            x,
            chunk_size,
            return_trajectory,
        )

    if name == "euler":
        if kwargs:
            raise TypeError(f"Unexpected Euler sampler kwargs: {sorted(kwargs)}.")
        if timesteps is None:
            raise ValueError("timesteps are required for the Euler sampler.")
        return sample_flow_euler(net, x, timesteps, return_trajectory)

    if name == "heun":
        if kwargs:
            raise TypeError(f"Unexpected Heun sampler kwargs: {sorted(kwargs)}.")
        if timesteps is None:
            raise ValueError("timesteps are required for the Heun sampler.")
        return sample_flow_heun(net, x, timesteps, return_trajectory)

    if name == "rk4":
        if kwargs:
            raise TypeError(f"Unexpected RK4 sampler kwargs: {sorted(kwargs)}.")
        if timesteps is None:
            raise ValueError("timesteps are required for the RK4 sampler.")
        return sample_flow_rk4(net, x, timesteps, return_trajectory)

    if name == "adaptive_heun":
        if "t_min" not in kwargs or "t_max" not in kwargs:
            raise ValueError("adaptive_heun requires t_min and t_max.")
        return sample_flow_adaptive_heun(net, x, t_min=kwargs.pop("t_min"), t_max=kwargs.pop("t_max"), return_trajectory=return_trajectory, **kwargs)

    raise ValueError(f"Unknown flow sampler {sampler!r}.")


@torch.inference_mode()
def _ddpm_eps_hat(model: Any, x: torch.Tensor, t: int, n_samples: int) -> torch.Tensor:
    t_idx = int(t) - 1
    t_norm = torch.full((n_samples, 1), float(t) / model._n_steps, device=model._device, dtype=model._fdtype)
    return model._get_eps_hat(x, t_norm, t_idx)


def _ddim_timesteps(model_n_steps: int, n_steps: int | None, device: torch.device) -> torch.Tensor:
    model_n_steps = int(model_n_steps)
    n_steps = model_n_steps if n_steps is None else int(n_steps)

    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    if n_steps > model_n_steps:
        raise ValueError(f"n_steps must be <= model._n_steps ({model_n_steps}), got {n_steps}.")
    if n_steps == model_n_steps:
        return torch.arange(model_n_steps, 0, -1, device=device, dtype=torch.long)
    if n_steps == 1:
        return torch.tensor([model_n_steps], device=device, dtype=torch.long)

    timesteps = torch.linspace(1, model_n_steps, n_steps, device=device)
    return timesteps.round().to(dtype=torch.long).flip(0)


@torch.inference_mode()
def sample_ddpm(
    model: Any,
    n_samples: int | None = None,
    return_trajectory: bool = True,
    sample_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Run the native reverse DDPM chain for a DDPM-like model."""
    model._net.eval()
    n_samples, x = model._resolve_sample_source(n_samples, sample_source)
    trajectory = [x] if return_trajectory else None

    for t in range(model._n_steps, 0, -1):
        t_idx = t - 1
        eps_hat = _ddpm_eps_hat(model, x, t, n_samples)
        scale = model._betas[t_idx] / torch.sqrt(1.0 - model._alpha_bar[t_idx])
        x = x - scale * eps_hat
        x = x / torch.sqrt(model._alphas[t_idx])

        if t > 1:
            x = x + model._sigma_max * model._sqrt_post_var[t_idx] * torch.randn_like(x)
        if trajectory is not None:
            trajectory.append(x)

    return x, trajectory


@torch.inference_mode()
def sample_ddim(
    model: Any,
    n_samples: int | None = None,
    n_steps: int | None = None,
    eta: float = 0.0,
    return_trajectory: bool = True,
    sample_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Run DDIM sampling for a DDPM-like model using epsilon predictions."""
    eta = float(eta)
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")

    model._net.eval()
    n_samples, x = model._resolve_sample_source(n_samples, sample_source)
    trajectory = [x] if return_trajectory else None
    timesteps = _ddim_timesteps(model._n_steps, n_steps, model._device)

    for i, t_value in enumerate(timesteps.tolist()):
        prev_t = 0 if i == timesteps.numel() - 1 else int(timesteps[i + 1].item())
        t_idx = int(t_value) - 1
        eps_hat = _ddpm_eps_hat(model, x, int(t_value), n_samples)

        alpha_t = model._alpha_bar[t_idx]
        alpha_prev = torch.ones((), device=model._device, dtype=model._fdtype) if prev_t == 0 else model._alpha_bar[prev_t - 1]
        one_minus_alpha_t = (1.0 - alpha_t).clamp_min(torch.finfo(model._fdtype).eps)
        x0_hat = (x - torch.sqrt(one_minus_alpha_t) * eps_hat) / torch.sqrt(alpha_t)

        sigma = eta * torch.sqrt(((1.0 - alpha_prev) / one_minus_alpha_t) * (1.0 - alpha_t / alpha_prev).clamp_min(0.0))
        direction_scale = (1.0 - alpha_prev - sigma.square()).clamp_min(0.0).sqrt()
        x = torch.sqrt(alpha_prev) * x0_hat + direction_scale * eps_hat

        if eta > 0.0 and prev_t > 0:
            x = x + sigma * torch.randn_like(x)
        if trajectory is not None:
            trajectory.append(x)

    return x, trajectory
