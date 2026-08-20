"""Tests for neural network modules."""

import inspect
import pytest
import torch
from gendynamics.nn import MLPModel, Transformer2DModel, TransformerModel, UNet2DModel, UNetModel
from gendynamics.diffusion import DDPMEps, DDPMV, DDPMX0
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def _ddpm(n_steps=10):
    return DDPMV(net=_ZeroNet(), dim=2, n_steps=n_steps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_private_timestep_embedding_shape_dtype_device(dtype, device):
    t = torch.rand(17, device=device, dtype=dtype)
    y = MLPModel._timestep_embedding(t, 32).to(dtype=dtype)
    assert y.shape == (17, 32)
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(("input_dim", "output_dim", "expected_shape"), [(2, None, (17, 2)), (3, 2, (17, 2))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_mlp_model_shape_dtype_device(input_dim, output_dim, expected_shape, dtype, device):
    model = MLPModel(input_dim=input_dim, output_dim=output_dim, width=16, depth=2, time_dim=8).to(device=device, dtype=dtype)
    x = torch.rand(17, input_dim, device=device, dtype=dtype)
    t = torch.rand(17, 1, device=device, dtype=dtype)
    y = model(x, t)
    assert y.shape == expected_shape
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()


def test_diffusers_image_models_import_and_unet_model_smoke():
    assert Transformer2DModel.__name__ == "Transformer2DModel"
    assert UNet2DModel.__name__ == "UNet2DModel"
    assert TransformerModel.__name__ == "TransformerModel"

    model = UNetModel(
        sample_size=8,
        n_steps=4,
        in_channels=1,
        out_channels=1,
        width=8,
        channel_mult=(1,),
        layers_per_block=1,
        norm_num_groups=1,
        attention=False,
    )
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([[0.0], [1.0]])
    y = model(x, t)

    assert y.shape == x.shape


def _tiny_transformer(**kwargs):
    params = {
        "sample_size": 8,
        "n_steps": 8,
        "in_channels": 3,
        "out_channels": 3,
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "patch_size": 2,
    }
    params.update(kwargs)
    return TransformerModel(**params)


def test_transformer_model_rgb_output_shape():
    model = _tiny_transformer()
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([[0.0], [1.0]])
    y = model(x, t)

    assert y.shape == x.shape


def test_transformer_model_multichannel_hrrr_like_output_shape():
    model = TransformerModel(
        sample_size=8,
        n_steps=8,
        in_channels=5,
        out_channels=5,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        patch_size=2,
    )
    x = torch.randn(2, 5, 8, 8)
    y = model(x, torch.tensor([0.25, 0.75]))

    assert y.shape == x.shape


def test_transformer_model_expands_scalar_timestep():
    model = _tiny_transformer(timestep_mode="discrete")
    timestep = model._prepare_timesteps(torch.tensor(3), batch_size=4, device=torch.device("cpu"))

    assert timestep.shape == (4,)
    assert timestep.dtype == torch.long
    assert torch.equal(timestep, torch.full((4,), 3, dtype=torch.long))


@pytest.mark.parametrize("t", [torch.tensor([0.0, 1.0]), torch.tensor([[0.0], [1.0]])])
def test_transformer_model_accepts_vector_timestep_shapes(t):
    model = _tiny_transformer()
    x = torch.randn(2, 3, 8, 8)
    y = model(x, t)

    assert y.shape == x.shape


def test_transformer_model_rejects_invalid_timestep_batch_size():
    model = _tiny_transformer()
    x = torch.randn(2, 3, 8, 8)

    with pytest.raises(ValueError, match="Expected 2 timesteps"):
        model(x, torch.tensor([0.0, 0.5, 1.0]))


def test_transformer_model_rejects_invalid_spatial_dimensions():
    model = _tiny_transformer()
    x = torch.randn(2, 3, 8, 10)

    with pytest.raises(ValueError, match="Expected spatial shape"):
        model(x, torch.tensor([0.0, 1.0]))


def test_transformer_model_float32_forward_backward_finite():
    model = _tiny_transformer().to(dtype=torch.float32)
    x = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    target = torch.randn_like(x)
    y = model(x, torch.tensor([0.25, 0.75], dtype=torch.float32))
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]

    assert torch.isfinite(y).all()
    assert torch.isfinite(loss)
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_transformer_model_continuous_timesteps_remain_unrounded():
    model = _tiny_transformer(timestep_mode="continuous")
    t = torch.tensor([0.125, 0.875], dtype=torch.float32)
    prepared = model._prepare_timesteps(t, batch_size=2, device=torch.device("cpu"))

    assert prepared.dtype == torch.float32
    assert torch.equal(prepared, t)
    assert not torch.equal(prepared, prepared.round())


def test_transformer_model_discrete_integer_timestep_support():
    model = _tiny_transformer(timestep_mode="discrete")
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0, 7], dtype=torch.int64)
    y = model(x, t)

    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_transformer_model_cuda_fp16_autocast():
    model = _tiny_transformer().cuda()
    x = torch.randn(2, 3, 8, 8, device="cuda")
    t = torch.tensor([0.25, 0.75], device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x, t)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="CUDA bf16 is not supported.")
def test_transformer_model_cuda_bf16_autocast():
    model = _tiny_transformer().cuda()
    x = torch.randn(2, 3, 8, 8, device="cuda")
    t = torch.tensor([0.25, 0.75], device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, t)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_transformer_model_accepts_no_class_labels_or_text_conditions():
    signature = inspect.signature(TransformerModel.forward)
    assert list(signature.parameters) == ["self", "x", "t"]

    model = _tiny_transformer()
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.25, 0.75])

    with pytest.raises(TypeError):
        model(x, t, class_labels=torch.zeros(2, dtype=torch.long))
    with pytest.raises(TypeError):
        model(x, t, encoder_hidden_states=torch.randn(2, 4, 8))
    with pytest.raises(ValueError, match="unconditional"):
        _tiny_transformer(num_classes=10)
    with pytest.raises(ValueError, match="cross_attention_dim"):
        _tiny_transformer(cross_attention_dim=8)


def test_unet_model_accepts_benchmark_config_keys():
    model = UNetModel(
        sample_size=16,
        n_steps=8,
        in_channels=3,
        out_channels=3,
        model_channels=8,
        num_res_blocks=1,
        attention_resolutions=(4,),
        dropout=0.1,
        channel_mult=(1, 2, 2),
        conv_resample=True,
        dims=2,
        num_heads=2,
        use_scale_shift_norm=True,
        norm_num_groups=1,
    )
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([[0.0], [1.0]])
    y = model(x, t)

    assert y.shape == x.shape


# --- Base._check_t validation ---

@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (True, TypeError, "bool"),
        (0, ValueError, r"\[1,"),
        (11, ValueError, r"\[1,"),
        (-0.1, ValueError, r"\[0, 1\]"),
        (1.1, ValueError, r"\[0, 1\]"),
        (torch.tensor([1, 2, 3]), ValueError, "shape"),
        (torch.tensor([0.5, 0.5, 1.5, 0.5]), ValueError, r"\[0, 1\]"),
        (torch.tensor([1, 2, 0, 4], dtype=torch.int32), ValueError, r"\[1,"),
    ],
)
def test_check_t_rejects_invalid_inputs(value, error, match):
    with pytest.raises(error, match=match):
        _ddpm(n_steps=10)._check_t(value, 4)


def test_check_t_valid_int_returns_broadcast_tensor():
    t = _ddpm(n_steps=10)._check_t(5, 4)
    assert t.shape == (4,)
    assert (t == 5).all()


# --- Base._sample_source with explicit base tensor ---

@pytest.mark.parametrize(
    ("dim", "base", "n_samples", "expected_shape"),
    [
        (2, torch.tensor([1.0, 2.0]), 5, (5, 2)),
        ((1, 4, 4), torch.arange(16, dtype=torch.float32).reshape(1, 4, 4), 3, (3, 1, 4, 4)),
    ],
)
def test_sample_source_single_base_expands_to_n_samples(dim, base, n_samples, expected_shape):
    m = DDPMV(net=_ZeroNet(), dim=dim, n_steps=10, base_or_sample=base)
    samples = m._sample_source(n_samples)

    assert samples.shape == expected_shape
    assert torch.equal(samples, base.unsqueeze(0).expand(expected_shape))


def test_sample_source_1d_base_wrong_dim_raises():
    base = torch.tensor([1.0, 2.0, 3.0])  # dim 3, model expects dim 2
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, base_or_sample=base)
    with pytest.raises(ValueError, match="dim"):
        m._sample_source(4)


def test_sample_source_2d_base_samples_from_rows():
    base = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, base_or_sample=base)
    samples = m._sample_source(4)
    assert samples.shape == (4, 2)


def test_ddpm_accepts_image_shaped_batches():
    m = DDPMV(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10)
    x = torch.randn(3, 1, 4, 4)

    loss = m.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert m.sample(2).shape == (2, 1, 4, 4)


def test_ddpm_sigma_max_scales_default_source(monkeypatch):
    def _ones(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _ones)
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=3.0)

    assert torch.equal(m._sample_source_default(4), torch.full((4, 2), 3.0))


def test_ddpm_sigma_max_scales_reverse_posterior_noise(monkeypatch):
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=2, sigma_max=4.0)
    m._sample_source = lambda n_samples: torch.zeros(n_samples, 2, device=m._device, dtype=m._fdtype)
    m._alpha_bar = torch.full((2,), 0.5, device=m._device, dtype=m._fdtype)
    m._alphas = torch.ones(2, device=m._device, dtype=m._fdtype)
    m._betas = torch.zeros(2, device=m._device, dtype=m._fdtype)
    m._sqrt_post_var = torch.tensor([0.0, 2.0], device=m._device, dtype=m._fdtype)
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)

    samples = m.sample(3)

    assert torch.equal(samples, torch.full((3, 2), 8.0, device=m._device, dtype=m._fdtype))


def test_ddpm_rejects_invalid_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        DDPMV(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=0.0)


def test_ddpm_sample_dispatches_native_sampler():
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=3, sampler="ddpm")

    out, trajectory = m._sample(2, return_trajectory=True)

    assert out.shape == (2, 2)
    assert len(trajectory) == 4


def test_ddpm_sample_trajectory_and_restart(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = DDPMV(net=_ZeroNet(), dim=2, n_steps=4)
    source = torch.randn(3, 2)

    trajectory = model.sample_trajectory(sample_source=source)
    restarted = model.sample(sample_source=trajectory[:, 2], start_step=2)

    assert trajectory.shape == (3, 5, 2)
    assert torch.allclose(restarted, trajectory[:, -1])


def test_ddpm_guidance_uses_reverse_variance(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = DDPMV(net=_ZeroNet(), dim=2, n_steps=2)
    model._alpha_bar = torch.full((2,), 0.5)
    model._alphas = torch.ones(2)
    model._betas = torch.zeros(2)
    model._sqrt_post_var = torch.tensor([0.0, 0.5])
    times = []

    def guidance(x, time):
        times.append(time)
        return torch.ones_like(x)

    samples = model.sample(sample_source=torch.zeros(3, 2), guidance=guidance)

    assert torch.allclose(samples, torch.full((3, 2), 0.25))
    assert torch.equal(times[0], torch.zeros(3))
    assert torch.equal(times[1], torch.full((3,), 0.5))


def test_ddpm_restart_requires_a_state():
    with pytest.raises(ValueError, match="sample_source"):
        _ddpm(n_steps=4).sample(2, start_step=2)


def test_ddpm_sample_accepts_explicit_sample_source():
    sample_source = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=3, sampler="ddpm", fdtype=torch.float32)
    m._alpha_bar = torch.full((3,), 0.5, device=m._device, dtype=m._fdtype)
    m._alphas = torch.ones(3, device=m._device, dtype=m._fdtype)
    m._betas = torch.zeros(3, device=m._device, dtype=m._fdtype)
    m._sqrt_post_var = torch.zeros(3, device=m._device, dtype=m._fdtype)

    out, trajectory = m._sample(return_trajectory=True, sample_source=sample_source)

    expected_source = sample_source.to(dtype=torch.float32)
    assert torch.allclose(trajectory[0], expected_source)
    assert torch.allclose(out, expected_source)
    assert torch.allclose(m.sample(sample_source=sample_source), expected_source)


def test_ddpm_sample_source_must_match_shape_and_count():
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=3)

    with pytest.raises(ValueError, match="sample_source"):
        m.sample(sample_source=torch.zeros(2, 3))

    with pytest.raises(ValueError, match="n_samples"):
        m._sample(3, sample_source=torch.zeros(2, 2))


def test_ddpm_uses_constructor_sampler_by_default():
    base = torch.zeros(2, dtype=torch.float32)
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=8, base_or_sample=base, sampler="ddim", sample_steps=4, eta=0.0)

    out, trajectory = m._sample(3, return_trajectory=True)

    assert out.shape == (3, 2)
    assert len(trajectory) == 5


def test_ddpm_sample_rejects_unknown_sampler():
    with pytest.raises(ValueError, match="ddpm"):
        DDPMV(net=_ZeroNet(), dim=2, n_steps=3, sampler="heun")


def test_ddpm_eps_loss_and_sample_smoke():
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=5)
    x = torch.randn(4, 2)

    loss = m.loss(x, t=0.5)
    out = m.sample(3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert out.shape == (3, 2)


@pytest.mark.parametrize("model_cls", [DDPMEps, DDPMV, DDPMX0])
def test_ddim_sampling_supports_all_ddpm_parameterizations(model_cls):
    base = torch.zeros(2, dtype=torch.float32)
    m = model_cls(net=_ZeroNet(), dim=2, n_steps=8, base_or_sample=base, sampler="ddim", sample_steps=4, eta=0.0)

    out, trajectory = m._sample(3, return_trajectory=True)

    assert out.shape == (3, 2)
    assert len(trajectory) == 5


def test_ddim_sample_accepts_explicit_sample_source():
    sample_source = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=8, sampler="ddim", sample_steps=4, eta=0.0, fdtype=torch.float32)
    m._alpha_bar = torch.ones(8, device=m._device, dtype=m._fdtype)

    out, trajectory = m._sample(return_trajectory=True, sample_source=sample_source)

    expected_source = sample_source.to(dtype=torch.float32)
    assert torch.allclose(trajectory[0], expected_source)
    assert torch.allclose(out, expected_source)
    assert torch.allclose(m.sample(sample_source=sample_source), expected_source)


def test_ddim_eta_zero_is_deterministic_with_fixed_source():
    base = torch.ones(2, dtype=torch.float32)
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=8, base_or_sample=base, sampler="ddim", sample_steps=4, eta=0.0)

    torch.manual_seed(0)
    out0 = m.sample(3)
    torch.manual_seed(1)
    out1 = m.sample(3)

    assert torch.equal(out0, out1)


def test_ddim_eta_positive_is_stochastic_with_fixed_source():
    base = torch.ones(2, dtype=torch.float32)
    m = DDPMEps(net=_ZeroNet(), dim=2, n_steps=8, base_or_sample=base, sampler="ddim", sample_steps=4, eta=1.0)

    torch.manual_seed(0)
    out0 = m.sample(3)
    torch.manual_seed(1)
    out1 = m.sample(3)

    assert not torch.equal(out0, out1)


def test_ddim_rejects_invalid_n_steps_and_eta():
    with pytest.raises(ValueError, match="sample_steps"):
        DDPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="ddim", sample_steps=0)
    with pytest.raises(ValueError, match="sample_steps"):
        DDPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="ddim", sample_steps=5)
    with pytest.raises(ValueError, match="eta"):
        DDPMEps(net=_ZeroNet(), dim=2, n_steps=4, sampler="ddim", eta=-1.0)
