"""Tests for optional third-party adapters."""

import pytest
import torch
from gendynamics.nn import MLPModel
from gendynamics.thirdparty import DLPMEpsOrigin, FlowMatchingOrigin, ScoreSDEOrigin, TEDMOrigin
from .utils import (
    _make_dlpm_vendor,
    _make_flow_matching_vendor,
    _make_score_sde_vendor,
    _make_tedm_vendor,
    _make_tedm_vendor_package_layout,
)


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


def _run_smoke(model, *, dtype=torch.float32):
    x = torch.randn(6, 2, dtype=dtype)
    loss = model.loss(x)
    sample = model.sample(5)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert sample.shape == (5, 2)
    assert sample.dtype == dtype


@pytest.mark.parametrize("fdtype", [torch.float32])
def test_flow_matching_origin_loss_and_sample(tmp_path, fdtype):
    model = FlowMatchingOrigin(
        net=MLPModel(input_dim=2, width=8, depth=1),
        dim=2,
        n_steps=4,
        package_root=str(_make_flow_matching_vendor(tmp_path)),
        fdtype=fdtype,
    )
    _run_smoke(model, dtype=fdtype)


def test_score_sde_origin_loss_and_sample(tmp_path):
    model = ScoreSDEOrigin(
        net=MLPModel(input_dim=2, width=8, depth=1),
        dim=2,
        n_steps=4,
        package_root=str(_make_score_sde_vendor(tmp_path)),
        fdtype=torch.float32,
    )
    _run_smoke(model)


def test_dlpmeps_origin_still_accepts_legacy_dtype_alias(tmp_path):
    with pytest.warns(DeprecationWarning, match=r"`dtype` is deprecated.*removed in v0\.2"):
        model = DLPMEpsOrigin(
            net=MLPModel(input_dim=2, width=8, depth=1),
            dim=2,
            n_steps=4,
            authors_root=str(_make_dlpm_vendor(tmp_path)),
            dtype=torch.float32,
        )
    _run_smoke(model)


def test_dlpmeps_origin_rejects_conflicting_fdtype_and_dtype():
    with pytest.raises(ValueError, match="Conflicting fdtype"):
        DLPMEpsOrigin(net=MLPModel(input_dim=2, width=8, depth=1), dim=2, fdtype=torch.float64, dtype=torch.float32)


def test_tedm_origin_loss_and_sample(tmp_path):
    model = TEDMOrigin(
        net=MLPModel(input_dim=2, width=8, depth=1),
        dim=2,
        n_steps=4,
        package_root=str(_make_tedm_vendor(tmp_path)),
        fdtype=torch.float32,
    )
    _run_smoke(model)


def test_tedm_origin_accepts_image_shaped_batches(tmp_path):
    model = TEDMOrigin(
        net=_ZeroNet(),
        dim=(1, 4, 4),
        n_steps=4,
        package_root=str(_make_tedm_vendor(tmp_path)),
        fdtype=torch.float32,
    )
    x = torch.randn(3, 1, 4, 4, dtype=torch.float32)

    loss = model.loss(x)
    sample = model.sample(2)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert sample.shape == (2, 1, 4, 4)


def test_tedm_origin_bypasses_vendor_package_initializers(tmp_path):
    model = TEDMOrigin(
        net=MLPModel(input_dim=2, width=8, depth=1),
        dim=2,
        n_steps=4,
        package_root=str(_make_tedm_vendor_package_layout(tmp_path)),
        fdtype=torch.float32,
    )
    _run_smoke(model)
