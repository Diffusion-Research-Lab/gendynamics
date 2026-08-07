"""Synthetic dataset registry and loading helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from ._noise import (
    sample_checker,
    sample_exponential,
    sample_gaussian,
    sample_scaled_isotropic_alpha_stable,
    sample_spiral,
    sample_student_t,
    sample_unbalanced_highdim_alpha_stable_mixture,
    sample_unbalanced_highdim_gaussian_mixture,
)

DatasetSampler = Callable[..., torch.Tensor]
SamplerKwargBuilders = dict[str, Callable[[dict[str, Any]], Any]]
_MISSING = object()


def _getpop(d: dict[str, Any], key: str, default: Any = _MISSING) -> Any:
    value = d.pop(key, _MISSING)
    if value is not _MISSING:
        return value
    if default is _MISSING:
        raise KeyError(key)
    return default


@dataclass(frozen=True)
class DatasetEntry:
    """Describe one synthetic dataset exposed through the public API."""

    name: str
    dataset_type: str
    description: str
    tail_index_alpha: Any = None
    split_mode: str = "random"
    standardize_default: bool = True
    dim: int | tuple[int, ...] | None = None
    n_samples: int | None = None
    sampler: DatasetSampler | None = None
    sampler_kwargs_builders: SamplerKwargBuilders = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tail_index_alpha": self.tail_index_alpha,
            "description": self.description,
            "split_mode": self.split_mode,
            "dataset_type": self.dataset_type,
            "dim": self.dim,
            "n_samples": self.n_samples,
        }


def _metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return value


def _copy_if_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(array, np.ndarray):
        return np.array(array, copy=True)
    return array


def _synthetic_entry(
    name: str,
    sampler: DatasetSampler,
    *,
    description: str,
    tail_index_alpha: Any = None,
    dim: int | None = None,
    sampler_kwargs_builders: SamplerKwargBuilders | None = None,
) -> DatasetEntry:
    return DatasetEntry(
        name=name,
        dataset_type="synthetic",
        description=description,
        tail_index_alpha=tail_index_alpha,
        split_mode="random",
        dim=dim,
        sampler=sampler,
        sampler_kwargs_builders=sampler_kwargs_builders or {},
    )


def _alpha_stable_mixture_base_scale(kwargs: dict[str, Any]) -> float:
    if "base_scale" in kwargs:
        return float(_getpop(kwargs, "base_scale"))
    return float(_getpop(kwargs, "base_std", 0.55))


def _resolve_dataset(target_data: str, all_datasets: dict[str, DatasetEntry]) -> DatasetEntry:
    key = target_data.lower()
    if key not in all_datasets:
        raise KeyError(f"Unknown dataset {target_data!r}. Available datasets: {sorted(all_datasets)}")
    return all_datasets[key]


def _standardize_split_arrays(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


def to_tensor_triplet(
    x_train: np.ndarray | torch.Tensor,
    x_val: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(_copy_if_numpy(x_train), device=device, dtype=dtype),
        torch.as_tensor(_copy_if_numpy(x_val), device=device, dtype=dtype),
        torch.as_tensor(_copy_if_numpy(x_test), device=device, dtype=dtype),
    )


def split_sample_indices(
    n_rows: int,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must lie in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    if n_rows < 3:
        raise ValueError("The dataset must contain at least 3 rows.")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be 'random' or 'chronological'.")

    if split_mode == "random":
        indices = np.arange(n_rows)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        return train_idx, val_idx, test_idx

    n_test = int(np.floor(test_size * n_rows))
    n_val = int(np.floor(val_size * n_rows))
    n_train = n_rows - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The requested val/test proportions leave an empty split.")
    return np.arange(0, n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n_rows)


def _split_frame_to_tensors(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any, Any]]:
    train_idx, val_idx, test_idx = split_sample_indices(
        len(frame),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    x_train = frame.iloc[train_idx].reset_index(drop=True).to_numpy()
    x_val = frame.iloc[val_idx].reset_index(drop=True).to_numpy()
    x_test = frame.iloc[test_idx].reset_index(drop=True).to_numpy()
    if standardize:
        x_train, x_val, x_test = _standardize_split_arrays(x_train, x_val, x_test)
    return to_tensor_triplet(x_train, x_val, x_test, device=device, dtype=dtype), (train_idx, val_idx, test_idx)


def _split_metadata(indices: Any) -> dict[str, Any]:
    return {"indices": [int(index) for index in indices], "n_samples": int(len(indices))}


def _build_return_metadata(
    *,
    entry: DatasetEntry,
    kind: str,
    target_data: str,
    params: dict[str, Any],
    split_config: dict[str, Any],
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
    split_indices: tuple[Any, Any, Any],
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = split_indices
    return {
        "dataset": entry.metadata(),
        "request": {
            "kind": kind,
            "name": target_data,
            "params": _metadata_value(params),
            "split": _metadata_value(split_config),
            "standardize": bool(standardize),
            "device": str(device),
            "dtype": str(dtype),
        },
        "loader": {},
        "splits": {
            "train": _split_metadata(train_idx),
            "val": _split_metadata(val_idx),
            "test": _split_metadata(test_idx),
        },
    }


SYNTHETIC_DATASETS: dict[str, DatasetEntry] = {
    "unbalanced_highdim_gaussian_mixture": _synthetic_entry(
        "unbalanced_highdim_gaussian_mixture",
        sample_unbalanced_highdim_gaussian_mixture,
        description="Imbalanced high-dimensional Gaussian mixture synthetic dataset.",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(_getpop(kwargs, "dim", 50)),
            "n_modes": lambda kwargs: int(_getpop(kwargs, "n_modes", 16)),
            "rank": lambda kwargs: int(_getpop(kwargs, "rank", 6)),
            "imbalance_tau": lambda kwargs: float(_getpop(kwargs, "imbalance_tau", 1.2)),
            "mean_scale": lambda kwargs: float(_getpop(kwargs, "mean_scale", 7.5)),
            "base_std": lambda kwargs: float(_getpop(kwargs, "base_std", 0.55)),
            "anisotropy": lambda kwargs: float(_getpop(kwargs, "anisotropy", 1.0)),
            "structure_seed": lambda kwargs: int(_getpop(kwargs, "structure_seed", 0)),
        },
    ),
    "unbalanced_highdim_alpha_stable_mixture": _synthetic_entry(
        "unbalanced_highdim_alpha_stable_mixture",
        sample_unbalanced_highdim_alpha_stable_mixture,
        description="Imbalanced high-dimensional mixture with alpha-stable local noise.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(_getpop(kwargs, "dim", 50)),
            "alpha": lambda kwargs: float(_getpop(kwargs, "alpha", 1.7)),
            "n_modes": lambda kwargs: int(_getpop(kwargs, "n_modes", 16)),
            "rank": lambda kwargs: int(_getpop(kwargs, "rank", 6)),
            "imbalance_tau": lambda kwargs: float(_getpop(kwargs, "imbalance_tau", 1.2)),
            "mean_scale": lambda kwargs: float(_getpop(kwargs, "mean_scale", 7.5)),
            "base_scale": _alpha_stable_mixture_base_scale,
            "anisotropy": lambda kwargs: float(_getpop(kwargs, "anisotropy", 1.0)),
            "structure_seed": lambda kwargs: int(_getpop(kwargs, "structure_seed", 0)),
        },
    ),
    "gaussian": _synthetic_entry(
        "gaussian",
        sample_gaussian,
        description="Isotropic Gaussian synthetic dataset.",
        tail_index_alpha=2.0,
        sampler_kwargs_builders={"dim": lambda kwargs: int(_getpop(kwargs, "dim", 1))},
    ),
    "checker": _synthetic_entry("checker", sample_checker, description="Checkerboard synthetic dataset.", dim=2),
    "spiral": _synthetic_entry(
        "spiral",
        sample_spiral,
        description="Noisy spiral synthetic dataset.",
        dim=2,
        sampler_kwargs_builders={
            "spiral_turns": lambda kwargs: float(_getpop(kwargs, "spiral_turns", 3.0)),
            "spiral_radius": lambda kwargs: float(_getpop(kwargs, "spiral_radius", 4.0)),
            "spiral_noise": lambda kwargs: float(_getpop(kwargs, "spiral_noise", 0.2)),
        },
    ),
    "alpha_stable": _synthetic_entry(
        "alpha_stable",
        sample_scaled_isotropic_alpha_stable,
        description="Isotropic alpha-stable synthetic dataset.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(_getpop(kwargs, "dim", 1)),
            "alpha": lambda kwargs: float(_getpop(kwargs, "alpha", 1.99)),
        },
    ),
    "student": _synthetic_entry(
        "student",
        sample_student_t,
        description="Student-t synthetic dataset.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(_getpop(kwargs, "dim", 1)),
            "nu": lambda kwargs: float(_getpop(kwargs, "nu", 10.0)),
        },
    ),
    "exponential": _synthetic_entry(
        "exponential",
        sample_exponential,
        description="Exponential synthetic dataset.",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(_getpop(kwargs, "dim", 1)),
            "rate": lambda kwargs: float(_getpop(kwargs, "rate", 1.0)),
        },
    ),
}


def fetch_synthetic_data(target_data: str, **kwargs: Any):
    return_metadata = bool(_getpop(kwargs, "return_metadata", False))
    entry = _resolve_dataset(target_data, SYNTHETIC_DATASETS)
    if entry.sampler is None:
        raise RuntimeError(f"Synthetic dataset {target_data!r} has no sampler.")

    n_samples = int(_getpop(kwargs, "n_samples", 10_000))
    val_size = float(_getpop(kwargs, "val_size", 0.15))
    test_size = float(_getpop(kwargs, "test_size", 0.15))
    random_state = int(_getpop(kwargs, "random_state", 0))
    standardize = bool(_getpop(kwargs, "standardize", False))
    device = _getpop(kwargs, "device", "cpu")
    dtype = _getpop(kwargs, "dtype", torch.float32)

    sampling_kwargs = {"n_samples": n_samples, "device": "cpu", "dtype": dtype}
    for name, builder in entry.sampler_kwargs_builders.items():
        sampling_kwargs[name] = builder(kwargs)

    samples = entry.sampler(**sampling_kwargs).detach().cpu().numpy()
    frame = pd.DataFrame(samples)
    tensors, split_indices = _split_frame_to_tensors(
        frame,
        split_mode=entry.split_mode,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        standardize=standardize,
        device=device,
        dtype=dtype,
    )
    if not return_metadata:
        return tensors
    metadata = _build_return_metadata(
        entry=entry,
        kind="synthetic",
        target_data=target_data,
        params=sampling_kwargs,
        split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
        standardize=standardize,
        device=device,
        dtype=dtype,
        split_indices=split_indices,
    )
    return (*tensors, metadata)


def get_dataset_metadata(target_data: str) -> dict[str, Any]:
    return _resolve_dataset(target_data, SYNTHETIC_DATASETS).metadata()


def list_datasets() -> list[str]:
    return sorted(SYNTHETIC_DATASETS)


__all__ = ["fetch_synthetic_data", "get_dataset_metadata", "list_datasets"]
