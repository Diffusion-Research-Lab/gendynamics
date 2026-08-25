"""Metrics module unittests."""

import numpy as np
import pytest
import torch
from gendynamics import DDPMV, DLPMEps, GaussianFlowEDM, GaussianFlowLinear
from gendynamics.metrics import classifier_tv_lower_bound, mmd_rbf, model_est_err_curve, model_est_jacobian_spectral_curve, sliced_wasserstein, tail_coverage_error
from .utils import _devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 3, device=device, dtype=dtype)

    sw2 = sliced_wasserstein(x, x, n_projections=64, n_grid=512, seed=123)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein_translation_formula(device, dtype):
    torch.manual_seed(0)
    n, d = 1024, 4
    x = torch.randn(n, d, device=device, dtype=dtype)
    shift = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + shift

    sw2 = sliced_wasserstein(x, y, n_projections=256, n_grid=1024, seed=123)
    expected = (shift @ shift).item() / float(d)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(expected, rel=0.15, abs=1e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_near_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 2, device=device, dtype=dtype)

    score = mmd_rbf(x, x)

    assert isinstance(score, float)
    assert abs(score) < 5e-3


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_detects_shifted_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 2, device=device, dtype=dtype)
    y = x + 2.0

    score = mmd_rbf(x, y)

    assert isinstance(score, float)
    assert score > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_supports_biased_and_unbiased_estimators(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(256, 2, device=device, dtype=dtype)
    y = x + 0.5

    score_biased = mmd_rbf(x, y, estimator="biased")
    score_unbiased = mmd_rbf(x, y, estimator="unbiased")

    assert isinstance(score_biased, float)
    assert isinstance(score_unbiased, float)
    assert score_biased >= 0.0
    assert torch.isfinite(torch.tensor(score_unbiased)).item()


def test_classifier_tv_lower_bound_near_zero_for_identical_distributions():
    torch.manual_seed(0)
    x = torch.randn(128, 2)

    score = classifier_tv_lower_bound(x, x, hidden_dim=16, n_folds=4, epochs=25, lr=5e-3, seed=123)

    assert isinstance(score, float)
    assert 0.0 <= score <= 0.1


def test_classifier_tv_lower_bound_detects_separated_distributions():
    torch.manual_seed(0)
    x_ref = torch.randn(128, 2) - 3.0
    x_gen = torch.randn(128, 2) + 3.0

    score = classifier_tv_lower_bound(x_ref, x_gen, hidden_dim=16, n_folds=4, epochs=40, lr=5e-3, seed=123)

    assert isinstance(score, float)
    assert score > 0.8


def test_classifier_tv_lower_bound_is_deterministic_for_fixed_seed():
    torch.manual_seed(0)
    x_ref = torch.randn(96, 3)
    x_gen = torch.randn(96, 3) + 0.5

    score_1 = classifier_tv_lower_bound(x_ref, x_gen, hidden_dim=12, n_folds=3, epochs=20, lr=3e-3, seed=7)
    score_2 = classifier_tv_lower_bound(x_ref, x_gen, hidden_dim=12, n_folds=3, epochs=20, lr=3e-3, seed=7)

    assert score_1 == pytest.approx(score_2, abs=0.0)


def test_classifier_tv_lower_bound_ignores_global_default_dtype():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        x_ref = torch.randn(32, 2, dtype=torch.float32)
        x_gen = torch.randn(32, 2, dtype=torch.float32) + 0.5

        score = classifier_tv_lower_bound(x_ref, x_gen, hidden_dim=8, n_folds=2, epochs=3, lr=1e-2, seed=5)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_classifier_tv_lower_bound_validates_inputs():
    x_ref = torch.randn(8, 2)
    x_gen = torch.randn(8, 3)

    with pytest.raises(ValueError, match="same feature dimension"):
        classifier_tv_lower_bound(x_ref, x_gen, n_folds=2, epochs=1)
    with pytest.raises(ValueError, match="n_folds"):
        classifier_tv_lower_bound(x_ref, x_ref, n_folds=1, epochs=1)
    with pytest.raises(ValueError, match="n_folds"):
        classifier_tv_lower_bound(x_ref[:2], x_ref[:2], n_folds=3, epochs=1)
    with pytest.raises(ValueError, match="finite"):
        classifier_tv_lower_bound(torch.tensor([[float("nan")], [0.0]]), torch.zeros(2, 1), n_folds=2, epochs=1)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_near_zero_for_identical_distributions(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(2048, 3, device=device, dtype=dtype)

    score = tail_coverage_error(x, x)

    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_detects_heavier_tailed_generated(device, dtype):
    torch.manual_seed(0)
    x_ref = torch.randn(2048, 2, device=device, dtype=dtype)
    x_gen = 3.0 * torch.randn(2048, 2, device=device, dtype=dtype)

    score = tail_coverage_error(x_ref, x_gen, tail="upper")

    assert isinstance(score, float)
    assert score > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_reduction_none_returns_tensor_per_prob(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 2, device=device, dtype=dtype)
    probs = [0.9, 0.95, 0.99]

    result = tail_coverage_error(x, x, probs=probs, reduction="none")

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == len(probs)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_lower_tail_detects_lighter_left_tail(device, dtype):
    torch.manual_seed(0)
    x_ref = torch.randn(2048, 2, device=device, dtype=dtype)
    x_gen = x_ref + 5.0

    score = tail_coverage_error(x_ref, x_gen, tail="lower")

    assert isinstance(score, float)
    assert score > 0.0


class _LinearTimeNet(torch.nn.Module):
    """Simple linear net with a known Lipschitz constant."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * x + 0.0 * t


class _DummyGenModel:
    """Minimal metric-compatible generator stub."""

    def __init__(self, family: str, n_steps: int, device: str, dtype: torch.dtype, scale: float) -> None:
        self._family = family
        self._n_steps = int(n_steps)
        self._fdtype = dtype
        self._idtype = torch.int64
        self._device = device
        self._net = _LinearTimeNet(scale=scale).to(device=device, dtype=dtype)

    def _sample_source(self, n_samples: int) -> torch.Tensor:
        return torch.zeros((n_samples, 1), device=self._device, dtype=self._fdtype)

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        if z is None:
            z = self._sample_source(len(x))
        if t is None:
            if self._family == "flow":
                t = torch.zeros((len(x), 1), device=self._device, dtype=self._fdtype)
            else:
                t = torch.ones((len(x),), device=self._device, dtype=self._idtype)
        return (x.square().mean() + z.square().mean() + 0.0 * t.to(dtype=x.dtype).mean()).to(dtype=x.dtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        return 3.0 * self._loss(x, z, t).mean()


class _PointwiseLossGenModel(_DummyGenModel):
    """Metric-compatible generator stub whose native loss is pointwise."""

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        return x.square()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("family", "expected_t"),
    [
        ("flow", np.arange(5)),
        ("diffusion", np.arange(1, 6)),
    ],
)
def test_model_est_jacobian_spectral_curve_matches_linear_constant(device, family, expected_t):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=5, device=device, dtype=torch.float64, scale=2.5)

    curve = model_est_jacobian_spectral_curve(model, x, n_power_iter=4)

    assert curve.shape == (5,)
    assert np.allclose(curve, 2.5)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["flow", "diffusion"])
def test_model_est_jacobian_spectral_curve_caps_time_grid_by_default(device, family):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=32, device=device, dtype=torch.float64, scale=2.5)

    curve = model_est_jacobian_spectral_curve(model, x, n_power_iter=2)

    assert curve.shape == (10,)
    assert np.allclose(curve, 2.5)


@pytest.mark.parametrize("device", _devices())
def test_model_est_jacobian_spectral_curve_validates_arguments(device):
    x = torch.randn(4, 1, device=device, dtype=torch.float32)
    model = _DummyGenModel(family="flow", n_steps=4, device=device, dtype=torch.float32, scale=1.0)

    with pytest.raises(ValueError, match="n_power_iter"):
        model_est_jacobian_spectral_curve(model, x, n_power_iter=0)
    with pytest.raises(ValueError, match="max_n_steps"):
        model_est_jacobian_spectral_curve(model, x, max_n_steps=0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(("family",), [("flow",), ("diffusion",)])
def test_model_est_err_curve_returns_native_grid(device, family):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=4, device=device, dtype=torch.float64, scale=1.0)

    curve = model_est_err_curve(model, x)

    assert curve.shape == (4,)
    assert np.all(curve >= 0.0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["flow", "diffusion"])
def test_model_est_err_curve_reduces_pointwise_loss(device, family):
    x = torch.tensor([[1.0, -2.0], [0.5, 1.5]], device=device, dtype=torch.float64)
    model = _PointwiseLossGenModel(family=family, n_steps=12, device=device, dtype=torch.float64, scale=1.0)

    curve = model_est_err_curve(model, x)

    assert curve.shape == (10,)
    assert np.allclose(curve, float(model.loss(x).item()))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (GaussianFlowLinear, {}),
        (GaussianFlowEDM, {}),
        (DDPMV, {}),
        (DLPMEps, {"alpha": 1.6}),
    ],
)
def test_model_est_err_curve_supports_native_and_mse_loss(device, model_cls, kwargs):
    x = torch.randn(8, 1, device=device, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(device=device, dtype=torch.float64)
    model = model_cls(net=net, dim=1, n_steps=8, fdtype=torch.float64, device=device, **kwargs)

    native_curve = model_est_err_curve(model, x, max_n_steps=5, loss_type="native")
    mse_curve = model_est_err_curve(model, x, max_n_steps=5, loss_type="mse")

    assert native_curve.shape == (5,)
    assert mse_curve.shape == (5,)
    assert np.all(native_curve >= 0.0)
    assert np.all(mse_curve >= 0.0)


def test_model_est_err_curve_rejects_unknown_loss_type():
    x = torch.randn(8, 1, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(dtype=torch.float64)
    model = GaussianFlowLinear(net=net, dim=1, n_steps=8, fdtype=torch.float64, device="cpu")

    with pytest.raises(ValueError, match="loss_type"):
        model_est_err_curve(model, x, loss_type="other")
