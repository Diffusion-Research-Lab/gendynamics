"""Training module unittests."""

import pytest
import torch
from gendynamics.training import train, _save_ckpt


class _DummyGenerativeModel:
    def __init__(self, dim: int):
        self._net = torch.nn.Sequential(
            torch.nn.Linear(dim, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, dim),
        )

    def loss(self, x, z=None):
        y = self._net(x)
        return ((y - x) ** 2).mean()


def _run_train(x, **kwargs):
    defaults = {
        "batch_size": 24,
        "n_epochs": 2,
        "lr": 1e-3,
        "device": "cpu",
        "num_workers": 0,
        "lr_schedule": "constant",
        "freq_logging": 10,
    }
    defaults.update(kwargs)
    return train(_DummyGenerativeModel(dim=x.shape[1]), target_data=x, **defaults)


def _ckpt_parts():
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
    return net, opt, sched


def test_train_returns_lightweight_stats_by_default():
    x = torch.randn(96, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=4)

    assert set(diagnostics) == {"train_config", "stats"}
    assert diagnostics["train_config"]["data_device"] == "cpu"
    assert diagnostics["train_config"]["log_grad_norm"] is False

    stats = diagnostics["stats"]
    assert stats["epoch"] == [1, 2, 3, 4]
    assert len(stats["training_loss"]) == 4
    assert "grad_norm" not in stats
    assert torch.isfinite(torch.tensor(stats["training_loss"])).all().item()


def test_train_stats_frequency_records_final_epoch():
    x = torch.randn(96, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=5, stats_freq_epochs=2)

    assert diagnostics["stats"]["epoch"] == [2, 4, 5]
    assert len(diagnostics["stats"]["training_loss"]) == 3


def test_train_can_log_grad_norm_when_requested():
    x = torch.randn(96, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=3, log_grad_norm=True)

    grad_norm = diagnostics["stats"]["grad_norm"]
    assert len(grad_norm) == 3
    assert all(v >= 0.0 for v in grad_norm)
    assert torch.isfinite(torch.tensor(grad_norm)).all().item()


def test_train_accepts_yaml_numeric_strings():
    x = torch.randn(24, 3, dtype=torch.float32)
    _, diagnostics = _run_train(
        x,
        batch_size="12",
        n_epochs="1",
        lr="5e-4",
        num_workers="0",
        weight_decay="1e-6",
        grad_clip_norm="1.0",
        warmup_steps="0",
        cosine_eta_min_ratio="0.05",
        freq_logging="1",
        stats_freq_epochs="1",
        log_grad_norm="false",
        ckpt_freq_epochs="1",
        ckpt_keep_last="1",
    )

    config = diagnostics["train_config"]
    assert config["batch_size"] == 12
    assert config["n_epochs"] == 1
    assert config["lr"] == pytest.approx(0.0005)
    assert config["weight_decay"] == pytest.approx(0.000001)
    assert config["grad_clip_norm"] == pytest.approx(1.0)
    assert config["stats_freq_epochs"] == 1
    assert config["log_grad_norm"] is False


def test_train_tiny_dataset_with_large_batch_still_trains():
    x = torch.randn(3, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, batch_size=16)

    assert len(diagnostics["stats"]["training_loss"]) == 2
    assert torch.isfinite(torch.tensor(diagnostics["stats"]["training_loss"])).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA data-device contract")
def test_train_preserves_cuda_target_data_by_default():
    x = torch.randn(16, 2, device="cuda", dtype=torch.float32)
    _, diagnostics = train(
        _DummyGenerativeModel(dim=2),
        target_data=x,
        batch_size=8,
        n_epochs=1,
        lr=1e-3,
        device="cuda",
        lr_schedule="constant",
    )

    assert diagnostics["train_config"]["data_device"].startswith("cuda")


def test_train_empty_dataset_raises_clear_error():
    x = torch.empty(0, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="target_data is empty"):
        _run_train(x, batch_size=8, n_epochs=1)


def test_train_rejects_weight_decay_without_adamw():
    gm = _DummyGenerativeModel(dim=2)
    x = torch.randn(8, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="weight_decay"):
        train(
            gm,
            target_data=x,
            batch_size=4,
            n_epochs=1,
            lr=1e-3,
            device="cpu",
            num_workers=0,
            use_adamw=False,
            weight_decay=1e-4,
        )


def test_train_rejects_non_tensor_target_data():
    gm = _DummyGenerativeModel(dim=2)

    with pytest.raises(TypeError, match="target_data must be a torch.Tensor"):
        train(gm, target_data=[[1.0, 2.0]], batch_size=1, n_epochs=1, device="cpu")


def test_train_rejects_source_with_wrong_dimension():
    gm = _DummyGenerativeModel(dim=2)
    x = torch.randn(8, 2, dtype=torch.float32)
    z = torch.randn(8, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="source_data dim"):
        train(gm, target_data=x, source_data=z, batch_size=4, n_epochs=1, device="cpu")


@pytest.mark.parametrize("kwargs, match", [
    ({"stats_freq_epochs": 0}, "stats_freq_epochs"),
    ({"lr_schedule": "bad_schedule"}, "Unknown lr_schedule"),
])
def test_train_rejects_invalid_options(kwargs, match):
    x = torch.randn(8, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match=match):
        _run_train(x, **kwargs)


def test_save_ckpt_writes_file_and_symlink(tmp_path):
    _save_ckpt(tmp_path, 3, 1, 10, 0.5, *_ckpt_parts(), {"lr": 1e-3})

    assert (tmp_path / "ckpt_epoch_0001.pt").exists()
    assert (tmp_path / "ckpt_last.pt").exists()
    ckpt = torch.load(tmp_path / "ckpt_epoch_0001.pt", map_location="cpu")
    assert ckpt["epoch"] == 1
    assert ckpt["loss"] == pytest.approx(0.5)


def test_save_ckpt_rotates_old_checkpoints_beyond_keep_last(tmp_path):
    net, opt, sched = _ckpt_parts()
    for epoch in range(1, 5):
        _save_ckpt(tmp_path, 2, epoch, epoch * 10, 0.5, net, opt, sched, {})

    kept = sorted(tmp_path.glob("ckpt_epoch_*.pt"))
    assert len(kept) == 2
    assert kept[0].name == "ckpt_epoch_0003.pt"
    assert kept[1].name == "ckpt_epoch_0004.pt"


def test_train_warns_for_num_workers_on_tensor():
    x = torch.randn(10, 2)

    with pytest.warns(UserWarning, match="num_workers > 0 is ignored"):
        _run_train(x, batch_size=5, num_workers=2, n_epochs=1)


@pytest.mark.parametrize("kwargs, expected_epochs", [
    ({"n_epochs": 3, "lr_schedule": "cosine"}, 3),
    ({"n_epochs": 2, "grad_clip_norm": 1.0}, 2),
])
def test_train_optional_features_complete(kwargs, expected_epochs):
    x = torch.randn(50, 2, dtype=torch.float32)
    _, diagnostics = _run_train(x, **kwargs)
    assert len(diagnostics["stats"]["training_loss"]) == expected_epochs
