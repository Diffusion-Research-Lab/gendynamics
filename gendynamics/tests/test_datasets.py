"""Tests for synthetic dataset helpers."""

import pytest
import torch
from gendynamics.datasets import fetch_synthetic_data, get_dataset_metadata, list_datasets


def test_fetch_synthetic_data_supports_gaussian():
    x_train, x_val, x_test = fetch_synthetic_data(
        "gaussian",
        n_samples=20,
        dim=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float32,
    )

    assert x_train.shape == (12, 3)
    assert x_val.shape == (4, 3)
    assert x_test.shape == (4, 3)
    assert x_train.dtype == torch.float32


def test_fetch_synthetic_data_return_metadata_includes_dataset_and_splits():
    x_train, x_val, x_test, metadata = fetch_synthetic_data(
        "gaussian",
        n_samples=20,
        dim=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float32,
        return_metadata=True,
    )

    assert x_train.shape == (12, 3)
    assert x_val.shape == (4, 3)
    assert x_test.shape == (4, 3)
    assert metadata["dataset"] == get_dataset_metadata("gaussian")
    assert metadata["request"]["kind"] == "synthetic"
    assert metadata["request"]["params"]["n_samples"] == 20
    assert metadata["request"]["params"]["dim"] == 3
    assert sum(metadata["splits"][name]["n_samples"] for name in ("train", "val", "test")) == 20


def test_fetch_synthetic_data_supports_unbalanced_highdim_gaussian_mixture():
    x_train, x_val, x_test = fetch_synthetic_data(
        "unbalanced_highdim_gaussian_mixture",
        n_samples=40,
        dim=10,
        n_modes=8,
        rank=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (24, 10)
    assert x_val.shape == (8, 10)
    assert x_test.shape == (8, 10)
    assert x_train.dtype == torch.float64


def test_fetch_synthetic_data_supports_unbalanced_highdim_alpha_stable_mixture():
    x_train, x_val, x_test = fetch_synthetic_data(
        "unbalanced_highdim_alpha_stable_mixture",
        n_samples=40,
        dim=10,
        alpha=1.6,
        n_modes=8,
        rank=3,
        base_std=0.35,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (24, 10)
    assert x_val.shape == (8, 10)
    assert x_test.shape == (8, 10)
    assert x_train.dtype == torch.float64


def test_bimodal_gaussian_datasets_are_not_registered():
    assert "balanced_bimodal_gaussian" not in list_datasets()
    assert "unbalanced_bimodal_gaussian" not in list_datasets()
    with pytest.raises(KeyError, match="balanced_bimodal_gaussian"):
        get_dataset_metadata("balanced_bimodal_gaussian")
    with pytest.raises(KeyError, match="unbalanced_bimodal_gaussian"):
        fetch_synthetic_data("unbalanced_bimodal_gaussian", n_samples=8)


def test_fetch_synthetic_data_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="no_such_dataset"):
        fetch_synthetic_data("no_such_dataset")
