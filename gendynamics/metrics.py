"""Evaluation metric helpers."""

import math
from typing import Literal
import numpy as np
import torch
from .diffusion import DLPMEps, DDPMV, DDPMX0
from .flow_matching import GaussianFlowDDPM, GaussianFlowEDM, GaussianFlowLinear, GaussianFlowOTLinear

__all__ = [
    "classifier_tv_lower_bound",
    "model_est_err_curve",
    "model_est_jacobian_spectral_curve",
    "mmd_rbf",
    "sliced_wasserstein",
    "tail_coverage_error",
]


def _to_tensor(x, *, device=None, dtype=torch.float64):
    """Convert array-like input to a tensor on the requested device and dtype."""
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return x.to(device=device, dtype=dtype)


def _to_2d_tensor(x, *, device=None, dtype=torch.float64):
    """Normalize array-like input to a non-empty 2D tensor."""
    x = _to_tensor(x, device=device, dtype=dtype)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.numel() == 0 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("input must be non-empty")
    return x


def _validate_same_feature_dim(x_ref, x_gen):
    """Ensure both sample sets share the same feature dimension."""
    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")


def _rbf_kernel_matrix(x, y, gamma):
    """Evaluate an RBF kernel matrix."""
    return torch.exp(-float(gamma) * torch.cdist(x, y, p=2).pow(2))


def _median_heuristic_gamma(x_ref, x_gen):
    """Median-heuristic bandwidth for an RBF kernel."""
    z = torch.cat([x_ref, x_gen], dim=0)
    d2 = torch.cdist(z, z, p=2).pow(2)
    positive = d2[d2 > 0]
    if positive.numel() == 0:
        return 1.0
    median_d2 = torch.median(positive)
    return float(0.5 / median_d2.clamp_min(torch.finfo(z.dtype).eps).item())


def _default_tail_probs(n, *, device, dtype, min_exceedances=10):
    """Choose a small default tail-probability grid with enough exceedances."""
    if n < 2:
        raise ValueError("need at least two samples")
    if min_exceedances <= 0:
        raise ValueError("min_exceedances must be strictly positive")
    p_min = max(float(min_exceedances) / float(n), 1e-12)
    base = torch.tensor([1e-1, 5e-2, 1e-2, 5e-3, 1e-3], device=device, dtype=dtype)
    probs = base[base >= p_min]
    if probs.numel() == 0:
        probs = torch.tensor([p_min], device=device, dtype=dtype)
    return probs


def mmd_rbf(x_ref, x_gen, gamma=None, estimator="biased"):
    """Compute the empirical squared RBF-kernel MMD between two samples."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if min(int(x_ref.shape[0]), int(x_gen.shape[0])) < 2:
        raise ValueError("mmd_rbf requires at least two samples in each input.")

    if gamma is None:
        gamma = _median_heuristic_gamma(x_ref, x_gen)
    elif float(gamma) <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")

    k_xx = _rbf_kernel_matrix(x_ref, x_ref, gamma)
    k_yy = _rbf_kernel_matrix(x_gen, x_gen, gamma)
    k_xy = _rbf_kernel_matrix(x_ref, x_gen, gamma)

    if estimator == "biased":
        mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    elif estimator == "unbiased":
        n_ref, n_gen = x_ref.shape[0], x_gen.shape[0]
        mmd2 = (k_xx.sum() - torch.trace(k_xx)) / (n_ref * (n_ref - 1))
        mmd2 = mmd2 + (k_yy.sum() - torch.trace(k_yy)) / (n_gen * (n_gen - 1))
        mmd2 = mmd2 - 2.0 * k_xy.mean()
    else:
        raise ValueError("estimator must be 'biased' or 'unbiased'.")

    return float(mmd2.item())


def sliced_wasserstein(x_ref, x_gen, n_projections=128, n_grid=1000, eps=1e-12, seed=None):
    """Compute the sliced squared 2-Wasserstein distance."""
    n_grid = int(n_grid)
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if int(n_projections) < 1:
        raise ValueError("n_projections must be at least 1.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    d = x_ref.shape[1]
    generator = None if seed is None else torch.Generator(device=x_ref.device).manual_seed(seed)
    dirs = torch.randn(int(n_projections), d, device=x_ref.device, dtype=x_ref.dtype, generator=generator)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=1, keepdim=True).clamp_min(torch.finfo(x_ref.dtype).eps)
    proj_ref = x_ref @ dirs.T
    proj_gen = x_gen @ dirs.T
    q = torch.linspace(0.0, 1.0 - float(eps), n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref = torch.quantile(proj_ref, q, dim=0)
    q_gen = torch.quantile(proj_gen, q, dim=0)
    return float(torch.trapz((q_ref - q_gen).pow(2), q, dim=0).mean().item())


def classifier_tv_lower_bound(
    x_ref,
    x_gen,
    *,
    hidden_dim: int = 64,
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int | None = None,
) -> float:
    """Estimate a lower bound on TV from held-out balanced classification accuracy.

    The identity TV(P, Q) = 2 a* - 1 relates total variation to the optimal
    balanced accuracy for distinguishing P from Q. A learned finite classifier is
    not necessarily optimal, so this returns only a lower bound.
    """
    hidden_dim = int(hidden_dim)
    n_folds = int(n_folds)
    epochs = int(epochs)
    lr = float(lr)
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive.")

    x_ref = _to_2d_tensor(x_ref, dtype=torch.float32)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if not torch.isfinite(x_ref).all() or not torch.isfinite(x_gen).all():
        raise ValueError("x_ref and x_gen must contain only finite values.")

    x_ref = x_ref.detach().cpu()
    x_gen = x_gen.detach().cpu()
    n_ref, d = x_ref.shape
    n_gen = x_gen.shape[0]
    if n_folds > min(n_ref, n_gen):
        raise ValueError("n_folds cannot exceed the number of samples in either class.")

    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))

    ref_folds = torch.tensor_split(torch.randperm(n_ref, generator=generator), n_folds)
    gen_folds = torch.tensor_split(torch.randperm(n_gen, generator=generator), n_folds)
    correct_ref = correct_gen = total_ref = total_gen = 0

    for fold_idx in range(n_folds):
        ref_val = ref_folds[fold_idx]
        gen_val = gen_folds[fold_idx]
        ref_train = torch.cat([fold for i, fold in enumerate(ref_folds) if i != fold_idx])
        gen_train = torch.cat([fold for i, fold in enumerate(gen_folds) if i != fold_idx])

        x_train = torch.cat([x_ref[ref_train], x_gen[gen_train]], dim=0)
        y_train = torch.cat([
            torch.zeros(len(ref_train), 1, dtype=x_ref.dtype),
            torch.ones(len(gen_train), 1, dtype=x_ref.dtype),
        ], dim=0)
        x_val = torch.cat([x_ref[ref_val], x_gen[gen_val]], dim=0)

        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std

        model = torch.nn.Sequential(
            torch.nn.Linear(d, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(dtype=x_train.dtype)
        for module in model:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0), generator=generator)
                if module.bias is not None:
                    fan_in = module.weight.shape[1]
                    bound = 1.0 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(module.bias, -bound, bound, generator=generator)

        class_weight = torch.cat([
            torch.full((len(ref_train), 1), 0.5 / len(ref_train), dtype=x_train.dtype),
            torch.full((len(gen_train), 1), 0.5 / len(gen_train), dtype=x_train.dtype),
        ], dim=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            logits = model(x_train)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, reduction="none") * class_weight).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = (model(x_val) >= 0.0).reshape(-1)
        ref_pred = pred[:len(ref_val)]
        gen_pred = pred[len(ref_val):]
        correct_ref += int((~ref_pred).sum().item())
        correct_gen += int(gen_pred.sum().item())
        total_ref += len(ref_val)
        total_gen += len(gen_val)

    balanced_accuracy = 0.5 * (correct_ref / total_ref + correct_gen / total_gen)
    return max(0.0, min(1.0, 2.0 * balanced_accuracy - 1.0))


def _tail_coverage_curve(x_ref, x_gen, probs=None, tail="upper", min_exceedances=10):
    """Tail exceedance calibration at reference thresholds."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    if probs is None:
        probs = _default_tail_probs(min(int(x_ref.shape[0]), int(x_gen.shape[0])), device=x_ref.device,
                                    dtype=x_ref.dtype, min_exceedances=min_exceedances)
    else:
        probs = _to_2d_tensor(probs, device=x_ref.device, dtype=x_ref.dtype).reshape(-1)
        if torch.any((probs <= 0) | (probs >= 1)):
            raise ValueError("probs must satisfy 0 < probs < 1")

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")

    quantiles = 1.0 - probs if tail == "upper" else probs
    thresholds = torch.quantile(x_ref, quantiles, dim=0)
    if tail == "upper":
        ref_coverage = (x_ref.unsqueeze(0) > thresholds.unsqueeze(1)).double().mean(dim=1)
        gen_coverage = (x_gen.unsqueeze(0) > thresholds.unsqueeze(1)).double().mean(dim=1)
    else:
        ref_coverage = (x_ref.unsqueeze(0) < thresholds.unsqueeze(1)).double().mean(dim=1)
        gen_coverage = (x_gen.unsqueeze(0) < thresholds.unsqueeze(1)).double().mean(dim=1)

    return probs, ref_coverage, gen_coverage


def tail_coverage_error(x_ref, x_gen, probs=None, tail="upper", min_exceedances=10, mode="log", reduction="mean", eps=1e-12):
    """Aggregate marginal tail-coverage mismatch over tail probabilities and features."""
    _, ref_cov, gen_cov = _tail_coverage_curve(x_ref, x_gen, probs=probs, tail=tail,
                                               min_exceedances=min_exceedances)

    if mode == "log":
        err = (torch.log(gen_cov + float(eps)) - torch.log(ref_cov + float(eps))).abs()
    elif mode == "abs":
        err = (gen_cov - ref_cov).abs()
    else:
        raise ValueError("mode must be 'log' or 'abs'")

    if reduction == "mean":
        return float(err.mean().item())
    if reduction == "none":
        return err
    raise ValueError("reduction must be 'mean' or 'none'")


def _require_family(gen_model: object) -> str:
    """Return the model family after validating it is metric-compatible."""
    family = getattr(gen_model, "_family", None)
    if family is None:
        raise ValueError(f"model metric utilities require a '._family' tag, got {type(gen_model)}.")
    if family not in {"flow", "diffusion"}:
        raise ValueError(f"model metric utilities only support 'diffusion' or 'flow' models, got {family}.")
    return family


def _time_grid(gen_model: object, x: torch.Tensor, max_n_steps: int | None = None) -> torch.Tensor:
    """Return the sampled step grid used for model diagnostics."""
    family = _require_family(gen_model)
    if max_n_steps is not None and int(max_n_steps) < 1:
        raise ValueError("max_n_steps must be at least 1.")
    if isinstance(gen_model, GaussianFlowEDM):
        n_steps = gen_model._n_steps if max_n_steps is None else min(int(max_n_steps), gen_model._n_steps)
        sigma_min = float(max(gen_model._eps, torch.finfo(gen_model._fdtype).tiny))
        return torch.linspace(sigma_min, gen_model._sigma_max, n_steps, device=x.device, dtype=gen_model._fdtype)
    if family == "flow":
        t_grid = torch.arange(gen_model._n_steps, device=x.device, dtype=gen_model._idtype)
    else:
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)
    if max_n_steps is None or int(max_n_steps) >= len(t_grid):
        return t_grid
    idx = torch.linspace(0, len(t_grid) - 1, steps=int(max_n_steps), device=t_grid.device)
    return t_grid.index_select(0, torch.round(idx).to(dtype=torch.long))


def _jacobian_spectral_at_time(
    gen_model: object,
    x_eval: torch.Tensor,
    t_step: torch.Tensor,
    *,
    n_power_iter: int,
) -> torch.Tensor:
    """Estimate the maximum samplewise Jacobian spectral norm at one time."""
    family = _require_family(gen_model)
    if isinstance(gen_model, GaussianFlowEDM):
        t_batch = t_step.to(dtype=gen_model._fdtype).reshape(1, 1)

        def model_at_time(z: torch.Tensor) -> torch.Tensor:
            x = z.unsqueeze(0)
            t = t_batch.to(device=z.device, dtype=z.dtype)
            return gen_model.vector_field(x, t).squeeze(0)

    else:
        denom = float(max(gen_model._n_steps - 1, 1)) if family == "flow" else float(gen_model._n_steps)
        t_batch = (t_step.to(dtype=gen_model._fdtype) / denom).reshape(1, 1)

        def model_at_time(z: torch.Tensor) -> torch.Tensor:
            x = z.unsqueeze(0)
            t = t_batch.to(device=z.device, dtype=z.dtype)
            return gen_model._net(x, t).squeeze(0)

    sample_values = []
    for x_i in x_eval:
        v = torch.randn_like(x_i)
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)

        for _ in range(int(n_power_iter)):
            x_base = x_i.detach()
            x_var = x_base.requires_grad_(True)
            y = model_at_time(x_var)
            _, jv = torch.autograd.functional.jvp(model_at_time, x_base, v, create_graph=False)
            jt_j_v = torch.autograd.grad(y, x_var, grad_outputs=jv, retain_graph=False, create_graph=False)[0]
            v = jt_j_v / torch.linalg.vector_norm(jt_j_v).clamp_min(1e-12)

        _, jv = torch.autograd.functional.jvp(model_at_time, x_i.detach(), v, create_graph=False)
        sample_values.append(torch.linalg.vector_norm(jv))

    return torch.stack(sample_values).max()


def _mse_loss_at_batch(model, x: torch.Tensor, t=None) -> torch.Tensor:
    """Compute a plain prediction-target MSE for one native training batch."""
    if isinstance(model, (GaussianFlowLinear, GaussianFlowOTLinear, GaussianFlowDDPM, GaussianFlowEDM)):
        pred, target, _ = model._precompute_loss(x=x, z=None, t=t)
        return torch.nn.functional.mse_loss(pred, target)

    if isinstance(model, DDPMV):
        x_1, x_t, eps, t_norm, _, a_bar_t = model._latent(x_1=x, eps=None, t=t)
        target = torch.sqrt(a_bar_t) * eps - torch.sqrt(1.0 - a_bar_t) * x_1
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, target)

    if isinstance(model, DDPMX0):
        x_1, x_t, _, t_norm, _, _ = model._latent(x_1=x, eps=None, t=t)
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, x_1)

    if isinstance(model, DLPMEps):
        x_1 = x.to(device=model._device, dtype=model._fdtype)
        n = x_1.size(0)
        n_a = model._n_trial_A
        n_g = model._n_trial_G
        t_checked = model._check_t(t, n) if t is not None else torch.randint(
            1, model._n_steps, (n,), device=model._device, dtype=model._idtype
        )
        t_e = t_checked.view(1, 1, n).expand(n_a, n_g, n).reshape(-1)
        t_norm = (t_checked / model._n_steps).view(1, 1, n).expand(n_a, n_g, n).reshape(-1, 1)
        eps = model._sample_source_default(n, expand_trials=True)
        gamma_1_t = model._gamma_1_t.index_select(0, t_e).unsqueeze(-1)
        sigma_1_t = model._sigma_1_t.index_select(0, t_e).unsqueeze(-1)
        x_1_e = x_1.view(1, 1, n, model._dim).expand(n_a, n_g, n, model._dim).reshape(-1, model._dim)
        x_t = gamma_1_t * x_1_e + sigma_1_t * eps
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, eps)

    raise ValueError(f"MSE model metric loss is not implemented for {type(model).__name__}.")


@torch.no_grad()
def model_est_err_curve(
    gen_model: object,
    x: torch.Tensor,
    *,
    max_n_steps: int | None = 10,
    loss_type: Literal["native", "mse"] = "native",
) -> np.ndarray:
    """Evaluate the mean score/vector fields estimation error across times."""
    family = _require_family(gen_model)
    if loss_type not in {"native", "mse"}:
        raise ValueError(f"loss_type must be 'native' or 'mse', got {loss_type!r}.")

    x_target = x.to(device=gen_model._device, dtype=gen_model._fdtype)
    x_source = torch.randn_like(x_target) if isinstance(gen_model, GaussianFlowEDM) else gen_model._sample_source(len(x_target))
    t_grid = _time_grid(gen_model, x_target, max_n_steps=max_n_steps)
    was_training = gen_model._net.training
    gen_model._net.eval()

    if family == "flow" and not isinstance(gen_model, GaussianFlowEDM):
        t_grid = t_grid.float() / gen_model._n_steps

    try:
        if loss_type == "native":
            curve = torch.stack([gen_model.loss(x=x_target, z=x_source, t=t).detach() for t in t_grid])
        else:
            curve = torch.stack([_mse_loss_at_batch(gen_model, x_target, t=t).detach() for t in t_grid])
    finally:
        if was_training:
            gen_model._net.train()

    return curve.cpu().numpy()


def model_est_jacobian_spectral_curve(
    gen_model: object,
    x: torch.Tensor,
    *,
    max_n_steps: int | None = 10,
    n_power_iter: int = 8,
) -> np.ndarray:
    """Estimate the Jacobian spectral norm across times."""
    _ = _require_family(gen_model)
    if int(n_power_iter) < 1:
        raise ValueError("n_power_iter must be at least 1.")

    x_eval = x.to(device=gen_model._device, dtype=gen_model._fdtype)
    t_grid = _time_grid(gen_model, x_eval, max_n_steps=max_n_steps)
    was_training = gen_model._net.training
    gen_model._net.eval()

    try:
        curve = torch.stack([_jacobian_spectral_at_time(gen_model, x_eval, t_step, n_power_iter=n_power_iter) for t_step in t_grid])
    finally:
        if was_training:
            gen_model._net.train()

    return curve.cpu().numpy()
