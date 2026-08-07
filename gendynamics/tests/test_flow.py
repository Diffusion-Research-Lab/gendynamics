"""Tests for flow-matching models and samplers."""

import pytest
import torch
from gendynamics import GaussianFlowEDM as PublicGaussianFlowEDM
from gendynamics._schedules import flux_shifted_timesteps
from gendynamics.flow_matching import GaussianFlowDDPM, GaussianFlowEDM, GaussianFlowLinear, GaussianFlowOTLinear
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class _OnesNet(torch.nn.Module):
    def forward(self, x, t):
        assert t.shape == (x.size(0), 1)
        return torch.ones_like(x)


class _ConstantNet(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x, t):
        return self.value * torch.ones_like(x) + 0.0 * t.reshape(-1, *([1] * (x.ndim - 1)))


def _assert_finite_loss(model_cls, *, t, device, dtype, **kwargs):
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype, **kwargs)
    loss = model.loss(torch.randn(5, 2, dtype=dtype, device=device), t=t)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def _ones_flow(sampler="euler", **kwargs):
    return GaussianFlowLinear(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        base_or_sample=torch.zeros(2),
        sampler=sampler,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("t", [3, 0.5])
def test_gaussian_flow_linear_accepts_common_time_formats(t, device, dtype):
    _assert_finite_loss(GaussianFlowLinear, t=t, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("sigma", [None, 0.5, torch.full((5,), 0.5), torch.full((5, 1), 0.5)])
def test_gaussian_flow_edm_accepts_common_sigma_formats(sigma, device, dtype):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=device, dtype=dtype)
    _assert_finite_loss(GaussianFlowEDM, t=sigma, device=device, dtype=dtype)


def test_gaussian_flow_edm_is_publicly_exported():
    assert PublicGaussianFlowEDM is GaussianFlowEDM


def test_gaussian_flow_edm_rejects_invalid_config_and_sigma():
    with pytest.raises(ValueError, match="sigma_data"):
        GaussianFlowEDM(net=_ZeroNet(), dim=2, sigma_data=0.0)
    with pytest.raises(ValueError, match="p_std"):
        GaussianFlowEDM(net=_ZeroNet(), dim=2, p_std=0.0)

    model = GaussianFlowEDM(net=_ZeroNet(), dim=2, n_steps=10)
    x = torch.randn(5, 2)
    with pytest.raises(ValueError, match="positive"):
        model.loss(x, t=0.0)
    with pytest.raises(ValueError, match="shape"):
        model.loss(x, t=torch.ones(5, 2))


def test_gaussian_flow_edm_vector_field_matches_denoiser_formula():
    model = GaussianFlowEDM(net=_ConstantNet(0.25), dim=2, n_steps=4, sigma_data=0.5, fdtype=torch.float64)
    x_sigma = torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float64)
    sigma = torch.tensor([0.2, 1.0], dtype=torch.float64)

    denoised = model.denoise(x_sigma, sigma)
    vector_field = model.vector_field(x_sigma, sigma)
    expected = (x_sigma - denoised) / sigma[:, None]

    assert torch.allclose(vector_field, expected)


def test_gaussian_flow_edm_euler_sample_integrates_vector_field():
    model = GaussianFlowEDM(
        net=_ZeroNet(),
        dim=2,
        n_steps=4,
        sigma_max=1.0,
        sigma_data=0.5,
        base_or_sample=torch.ones(2, dtype=torch.float64),
        sampler="euler",
        sample_steps=1,
        fdtype=torch.float64,
    )

    out, trajectory = model._sample(3, return_trajectory=True)

    sigma_0 = torch.tensor(model._sigma_max, dtype=torch.float64)
    sigma_1 = torch.tensor(max(model._eps, torch.finfo(model._fdtype).tiny), dtype=torch.float64)
    factor = 1.0 + (sigma_1 - sigma_0) * sigma_0 / (sigma_0.square() + model.sigma_data**2)
    assert torch.allclose(out, factor.expand_as(out))
    assert len(trajectory) == 2


def test_gaussian_flow_edm_chunked_sample_matches_full_batch():
    model = GaussianFlowEDM(
        net=_ZeroNet(),
        dim=2,
        n_steps=4,
        sigma_max=1.0,
        sigma_data=0.5,
        sampler="euler",
        sample_steps=2,
    )
    sample_source = torch.randn(5, 2)

    full = model.sample(sample_source=sample_source)
    chunked = model.sample(sample_source=sample_source, chunk_size=2)

    assert torch.allclose(chunked, full)


@pytest.mark.parametrize(("ot_method", "ot_reg"), [("exact", 0.05), ("sinkhorn", 1.0)])
def test_gaussian_flow_ot_linear_produces_finite_loss(ot_method, ot_reg):
    model = GaussianFlowOTLinear(
        net=_ZeroNet(),
        dim=2,
        n_steps=10,
        device="cpu",
        fdtype=torch.float32,
        ot_method=ot_method,
        ot_reg=ot_reg,
    )

    loss = model.loss(torch.randn(5, 2), t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_gaussian_flow_ot_linear_rejects_invalid_ot_config():
    with pytest.raises(ValueError, match="OT method"):
        GaussianFlowOTLinear(net=_ZeroNet(), dim=2, ot_method="bad")
    with pytest.raises(ValueError, match="ot_reg"):
        GaussianFlowOTLinear(net=_ZeroNet(), dim=2, ot_reg=0.0)


def test_gaussian_flow_ot_linear_samples_low_cost_pairs():
    model = GaussianFlowOTLinear(net=_ZeroNet(), dim=1, n_steps=4, device="cpu", fdtype=torch.float32)
    x_0 = torch.tensor([[0.0], [10.0]])
    x_1 = torch.tensor([[0.1], [10.1]])

    coupled_0, coupled_1 = model._sample_ot_coupling(x_0, x_1)

    assert coupled_0.shape == x_0.shape
    assert coupled_1.shape == x_1.shape
    assert torch.all((coupled_1 - coupled_0).abs() < 1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_gaussian_flow_loss_rejects_invalid_t(device, dtype):
    with pytest.raises(ValueError):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype).loss(
            torch.randn(5, 2, dtype=dtype, device=device), t=11
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_gaussian_flow_ddpm_produces_finite_loss(device, dtype):
    _assert_finite_loss(GaussianFlowDDPM, t=0.5, device=device, dtype=dtype)


@pytest.mark.parametrize("model_cls", [GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowEDM])
def test_gaussian_flow_sigma_max_scales_default_source(model_cls, monkeypatch):
    def _ones(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _ones)
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=2.5, device="cpu", fdtype=torch.float32)

    assert torch.equal(model._sample_source_default(3), torch.full((3, 2), 2.5))


def test_gaussian_flow_rejects_invalid_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=0.0, device="cpu", fdtype=torch.float32)


@pytest.mark.parametrize("model_cls", [GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowEDM])
def test_gaussian_flow_accepts_image_shaped_batches(model_cls):
    model = model_cls(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10, device="cpu", fdtype=torch.float32)
    x = torch.randn(3, 1, 4, 4)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.sample(2).shape == (2, 1, 4, 4)


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (True, TypeError, "bool"),
        (11, ValueError, None),
        (torch.tensor([0.5, 1.5, 0.3, 0.4, 0.5]), ValueError, r"\[0, 1\]"),
    ],
)
def test_gaussian_flow_check_t_rejects_invalid_inputs(value, error, match):
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(error, match=match):
        m._check_t(value, 5)


def test_gaussian_flow_latent_x1_wrong_dim_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError, match="x1"):
        m._latent(torch.randn(4, 3))  # dim 3, model expects dim 2


def test_gaussian_flow_latent_x0_shape_mismatch_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    x1 = torch.randn(4, 2)
    x0 = torch.randn(4, 3)  # wrong dim
    with pytest.raises(ValueError, match="x0"):
        m._latent(x1, x_0=x0)


@pytest.mark.parametrize("sampler", ["euler", "heun", "rk4"])
def test_gaussian_flow_sample_dispatches_named_samplers(sampler):
    out, trajectory = _ones_flow(sampler)._sample(3, return_trajectory=True)

    assert torch.allclose(out, torch.ones(3, 2))
    assert len(trajectory) == 5


@pytest.mark.parametrize("model_cls", [GaussianFlowLinear, GaussianFlowOTLinear, GaussianFlowDDPM])
def test_non_edm_gaussian_flows_use_base_net_sampler(model_cls):
    model = model_cls(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.2,
        t_max=0.8,
        base_or_sample=torch.zeros(2),
        sampler="euler",
    )

    out, trajectory = model._sample(3, return_trajectory=True)

    assert torch.allclose(out, torch.full((3, 2), 0.6))
    assert len(trajectory) == 5


def test_gaussian_flow_uses_constructor_sampler_by_default():
    out, trajectory = _ones_flow("euler")._sample(3, return_trajectory=True)

    assert torch.allclose(out, torch.ones(3, 2))
    assert len(trajectory) == 5


def test_gaussian_flow_sample_accepts_explicit_sample_source():
    sample_source = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    model = GaussianFlowLinear(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        sampler="euler",
        fdtype=torch.float32,
    )

    out, trajectory = model._sample(return_trajectory=True, sample_source=sample_source)

    expected_source = sample_source.to(dtype=torch.float32)
    assert torch.allclose(trajectory[0], expected_source)
    assert torch.allclose(out, expected_source + 1.0)
    assert torch.allclose(model.sample(sample_source=sample_source), expected_source + 1.0)


@pytest.mark.parametrize("sampler", ["euler", "heun", "rk4"])
def test_gaussian_flow_chunked_sample_matches_full_batch(sampler):
    sample_source = torch.randn(5, 2)
    model = _ones_flow(sampler)

    full = model.sample(sample_source=sample_source)
    chunked = model.sample(sample_source=sample_source, chunk_size=2)

    assert torch.allclose(chunked, full)


def test_gaussian_flow_chunked_trajectory_matches_full_batch():
    sample_source = torch.randn(5, 2)
    model = _ones_flow("euler")

    full, full_trajectory = model._sample(sample_source=sample_source, return_trajectory=True)
    chunked, chunked_trajectory = model._sample(sample_source=sample_source, return_trajectory=True, chunk_size=2)

    assert torch.allclose(chunked, full)
    assert len(chunked_trajectory) == len(full_trajectory)
    for chunked_step, full_step in zip(chunked_trajectory, full_trajectory):
        assert torch.allclose(chunked_step, full_step)


def test_gaussian_flow_rejects_invalid_chunk_size():
    model = _ones_flow("euler")

    with pytest.raises(ValueError, match="chunk_size"):
        model.sample(2, chunk_size=0)


def test_gaussian_flow_adaptive_heun_rejects_chunk_size():
    model = _ones_flow("adaptive_heun", h_init=0.25)

    with pytest.raises(ValueError, match="chunk_size"):
        model.sample(3, chunk_size=2)


def test_gaussian_flow_sample_source_must_match_shape_and_count():
    model = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=4)

    with pytest.raises(ValueError, match="sample_source"):
        model.sample(sample_source=torch.zeros(2, 3))

    with pytest.raises(ValueError, match="n_samples"):
        model._sample(3, sample_source=torch.zeros(2, 2))


def test_gaussian_flow_sample_accepts_flux_shifted_schedule():
    model = GaussianFlowLinear(
        net=_ZeroNet(), dim=2, n_steps=4, base_or_sample=torch.zeros(2), sampler="heun",
        schedule="flux_shifted", image_seq_len=1024, sample_steps=32,
    )

    out = model.sample(2)

    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.zeros_like(out))


def test_gaussian_flow_rejects_sampler_and_schedule_aliases():
    with pytest.raises(ValueError, match="adaptive"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, sampler="adaptive")
    with pytest.raises(ValueError, match="flux"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, schedule="flux", image_seq_len=1024)


def test_flux_shifted_schedule_is_monotone_and_seq_len_dependent():
    small = flux_shifted_timesteps(4, 0.0, 1.0, image_seq_len=256)
    large = flux_shifted_timesteps(4, 0.0, 1.0, image_seq_len=4096)

    assert small[0].item() == 0.0
    assert small[-1].item() == 1.0
    assert torch.all(small[1:] > small[:-1])
    assert torch.all(large[1:] > large[:-1])
    assert large[1].item() > small[1].item()


def test_gaussian_flow_adaptive_heun_smoke():
    out = _ones_flow("adaptive_heun", h_init=0.25).sample(3)

    assert torch.allclose(out, torch.ones(3, 2))
