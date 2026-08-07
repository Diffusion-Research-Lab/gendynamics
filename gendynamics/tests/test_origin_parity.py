"""Parity tests between native models and origin adapters."""

import pytest
import torch
from gendynamics.diffusion import DDPMV, DDPMX0, DLPMEps
from gendynamics.flow_matching import GaussianFlowLinear
from gendynamics.thirdparty import DLPMEpsOrigin, FlowMatchingOrigin, ScoreSDEOrigin
from .utils import (
    _AffineTimeNet,
    _assert_grad_lists_close,
    _assert_tensors_close,
    _collect_grads,
    _make_flow_matching_vendor,
    _make_score_sde_vendor,
    _reset_seeds,
)


def _make_dlpm_pair():
    native = DLPMEps(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=8,
        alpha=1.9,
        n_trial_A=1,
        n_trial_G=1,
        reduce_type="median",
        fdtype=torch.float32,
        device="cpu",
    )
    origin = DLPMEpsOrigin(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=8,
        alpha=1.9,
        monte_carlo_outer=1,
        monte_carlo_inner=1,
        loss_monte_carlo="median",
        fdtype=torch.float32,
        device="cpu",
    )
    return native, origin


def _make_flow_pair(tmp_path):
    base = torch.tensor([0.4, -0.3], dtype=torch.float32)
    native = GaussianFlowLinear(net=_AffineTimeNet(), dim=2, n_steps=6, t_min=0.0, t_max=1.0, base_or_sample=base, fdtype=torch.float32)
    origin = FlowMatchingOrigin(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=6,
        package_root=str(_make_flow_matching_vendor(tmp_path, exact=True)),
        base_or_sample=base,
        fdtype=torch.float32,
    )
    return native, origin


def _assert_same_loss_and_grads(native, origin, x, seed: int):
    _reset_seeds(seed)
    native_loss = native.loss(x)
    native_grads = _collect_grads(native_loss, native._net)
    _reset_seeds(seed)
    origin_loss = origin.loss(x)
    origin_grads = _collect_grads(origin_loss, origin._net)
    _assert_tensors_close(native_loss.detach(), origin_loss.detach(), atol=1e-7, rtol=1e-7)
    _assert_grad_lists_close(native_grads, origin_grads, atol=2e-7, rtol=1e-6)


def test_dlpmeps_matches_origin_loss_and_gradients():
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(6, 2)
    native, origin = _make_dlpm_pair()
    for seed in (0, 1, 2):
        _assert_same_loss_and_grads(native, origin, x, seed)


def test_dlpmeps_matches_origin_sample_and_one_step_update():
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(6, 2)
    native, origin = _make_dlpm_pair()

    _reset_seeds(3)
    native_sample = native.sample(5)
    _reset_seeds(3)
    origin_sample = origin.sample(5)
    _assert_tensors_close(native_sample, origin_sample, atol=2e-5, rtol=1e-5)

    native_optim = torch.optim.SGD(native._net.parameters(), lr=0.05)
    origin_optim = torch.optim.SGD(origin._net.parameters(), lr=0.05)

    _reset_seeds(5)
    native_loss = native.loss(x)
    native_optim.zero_grad()
    native_loss.backward()
    native_optim.step()

    _reset_seeds(5)
    origin_loss = origin.loss(x)
    origin_optim.zero_grad()
    origin_loss.backward()
    origin_optim.step()

    for native_param, origin_param in zip(native._net.parameters(), origin._net.parameters()):
        _assert_tensors_close(native_param.detach(), origin_param.detach(), atol=2e-7, rtol=1e-6)


def test_gaussian_flow_linear_matches_flow_matching_origin(tmp_path, monkeypatch):
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)
    z = torch.tensor([[0.2, -1.0], [1.5, 0.3], [0.7, -0.4]], dtype=torch.float32)
    t = torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.float32)
    native, origin = _make_flow_pair(tmp_path)

    monkeypatch.setattr(torch, "rand", lambda size, device=None, dtype=None: t.squeeze(-1).to(device=device, dtype=dtype))
    origin_loss = origin.loss(x, z=z)
    monkeypatch.undo()
    native_loss = native.loss(x, z=z, t=t)

    _assert_tensors_close(native_loss.detach(), origin_loss.detach())
    _assert_grad_lists_close(_collect_grads(native_loss, native._net), _collect_grads(origin_loss, origin._net))
    _assert_tensors_close(native.sample(4), origin.sample(4))


def test_gaussian_flow_linear_matches_flow_matching_origin_after_one_step(tmp_path, monkeypatch):
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)
    z = torch.tensor([[0.2, -1.0], [1.5, 0.3], [0.7, -0.4]], dtype=torch.float32)
    t = torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.float32)
    native, origin = _make_flow_pair(tmp_path)

    native_optim = torch.optim.SGD(native._net.parameters(), lr=0.1)
    origin_optim = torch.optim.SGD(origin._net.parameters(), lr=0.1)
    monkeypatch.setattr(torch, "rand", lambda size, device=None, dtype=None: t.squeeze(-1).to(device=device, dtype=dtype))

    native_loss = native.loss(x, z=z, t=t)
    origin_loss = origin.loss(x, z=z)
    native_optim.zero_grad()
    native_loss.backward()
    native_optim.step()
    origin_optim.zero_grad()
    origin_loss.backward()
    origin_optim.step()
    monkeypatch.undo()

    for native_param, origin_param in zip(native._net.parameters(), origin._net.parameters()):
        _assert_tensors_close(native_param.detach(), origin_param.detach(), atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize(("native_cls", "mode"), [(DDPMV, "v"), (DDPMX0, "x0")])
def test_ddpm_variants_match_score_sde_origin_with_exact_vendor(tmp_path, native_cls, mode):
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)
    native = native_cls(net=_AffineTimeNet(), dim=2, n_steps=6, fdtype=torch.float32, device="cpu")
    origin = ScoreSDEOrigin(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=6,
        package_root=str(_make_score_sde_vendor(tmp_path, mode=mode)),
        fdtype=torch.float32,
        device="cpu",
    )

    for seed in (0, 1, 2):
        _assert_same_loss_and_grads(native, origin, x, seed)
        _reset_seeds(seed)
        native_sample = native.sample(5)
        _reset_seeds(seed)
        origin_sample = origin.sample(5)
        _assert_tensors_close(native_sample, origin_sample, atol=1e-6, rtol=1e-6)
