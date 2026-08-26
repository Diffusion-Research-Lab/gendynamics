"""Training utilities for generative models."""

import logging
import math
import warnings
from pathlib import Path
from typing import Any
import torch


def train(
    generative_model,
    target_data: torch.Tensor,
    source_data: torch.Tensor | None = None,
    batch_size: int = 512,
    n_epochs: int = 250,
    lr: float = 1e-4,
    device: torch.device = "cpu",
    data_device: torch.device | str | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 2,
    use_adamw: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    lr_schedule: str = "cosine",
    warmup_steps: int = 0,
    cosine_eta_min_ratio: float = 0.0,
    freq_logging: int = 10,
    stats_freq_epochs: int = 1,
    log_grad_norm: bool = False,
    ckpt_dir: str | None = None,
    ckpt_freq_epochs: int = 10,
    ckpt_keep_last: int = 3,
    validation_data: torch.Tensor | None = None,
    restore_best: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Train a native gendynamics generative model and return diagnostics."""
    logger = logging.getLogger(__name__)

    # Normalize scalar configuration first so errors are raised before setup work.
    batch_size = int(batch_size)
    n_epochs = int(n_epochs)
    lr = float(lr)
    num_workers = int(num_workers)
    weight_decay = float(weight_decay)
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    warmup_steps = int(warmup_steps)
    cosine_eta_min_ratio = float(cosine_eta_min_ratio)
    freq_logging = int(freq_logging)
    if freq_logging < 0:
        raise ValueError("freq_logging must be >= 0.")
    stats_freq_epochs = int(stats_freq_epochs)
    if stats_freq_epochs < 1:
        raise ValueError("stats_freq_epochs must be >= 1.")
    ckpt_freq_epochs = int(ckpt_freq_epochs)
    if ckpt_freq_epochs < 1:
        raise ValueError("ckpt_freq_epochs must be >= 1.")
    ckpt_keep_last = int(ckpt_keep_last)
    if isinstance(log_grad_norm, str):
        log_grad_norm = log_grad_norm.lower().strip() in {"1", "true", "yes", "y", "on"}
    else:
        log_grad_norm = bool(log_grad_norm)

    # Validate the model and data contract.
    if not hasattr(generative_model, "_net") or not isinstance(generative_model._net, torch.nn.Module):
        raise ValueError("generative_model must have a torch.nn.Module attribute `_net`.")
    if not hasattr(generative_model, "loss") or not callable(generative_model.loss):
        raise ValueError("generative_model must have a callable method `loss(x, z=...)`.")
    if not isinstance(target_data, torch.Tensor):
        raise TypeError(f"target_data must be a torch.Tensor, got {type(target_data)}")
    if source_data is not None and not isinstance(source_data, torch.Tensor):
        raise TypeError(f"source_data must be a torch.Tensor or None, got {type(source_data)}")
    if validation_data is not None and not isinstance(validation_data, torch.Tensor):
        raise TypeError(f"validation_data must be a torch.Tensor or None, got {type(validation_data)}")
    if restore_best and validation_data is None:
        raise ValueError("restore_best requires validation_data.")
    if not use_adamw and float(weight_decay) != 0.0:
        raise ValueError("weight_decay is only supported with AdamW in this trainer.")

    # Stage tensors without silently forcing them back to CPU.
    device = torch.device(device)
    data_device = None if data_device is None else torch.device(data_device)

    target = target_data.detach()
    if data_device is not None:
        target = target.to(device=data_device)
    target = target.contiguous()
    if target.size(0) == 0:
        raise ValueError("target_data is empty; at least one sample is required for training.")

    if source_data is None:
        source = None
    else:
        source = source_data.detach()
        if data_device is not None:
            source = source.to(device=data_device)
        source = source.contiguous()
        if source.size(0) == 0:
            raise ValueError("source_data is empty; at least one sample is required when provided.")
        if source.size(-1) != target.size(-1):
            raise ValueError(f"source_data dim {source.size(-1)} != target_data dim {target.size(-1)}")

    if validation_data is None:
        validation = None
    else:
        validation = validation_data.detach()
        if data_device is not None:
            validation = validation.to(device=data_device)
        validation = validation.contiguous()
        if validation.size(0) == 0:
            raise ValueError("validation_data is empty; at least one sample is required when provided.")
        if validation.size(-1) != target.size(-1):
            raise ValueError(f"validation_data dim {validation.size(-1)} != target_data dim {target.size(-1)}")

    dtype = target.dtype
    net = generative_model._net.to(device=device, dtype=dtype)
    net.train()

    # Build tensor batching. Full tensor targets are shuffled once per epoch.
    if pin_memory is None:
        pin = device.type == "cuda" and target.device.type == "cpu"
        source_pin = device.type == "cuda" and source is not None and source.device.type == "cpu"
    else:
        pin = bool(pin_memory) and device.type == "cuda" and target.device.type == "cpu"
        source_pin = bool(pin_memory) and device.type == "cuda" and source is not None and source.device.type == "cpu"

    if num_workers > 0:
        warnings.warn(
            "num_workers > 0 is ignored for in-memory tensor targets; set num_workers=0 to suppress this warning.",
            stacklevel=2,
        )

    target_pin = bool(pin and target.device.type == "cpu" and torch.cuda.is_available())
    validation_pin = bool(pin and validation is not None and validation.device.type == "cpu" and torch.cuda.is_available())
    steps_per_epoch = (len(target) + batch_size - 1) // batch_size

    # Optimizer and learning-rate schedule.
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=lr, weight_decay=weight_decay)

    main_steps = max(1, n_epochs * steps_per_epoch - warmup_steps)
    if lr_schedule in (None, "none", "constant"):
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=main_steps)
    elif lr_schedule == "cosine":
        if not 0.0 <= cosine_eta_min_ratio < 1.0:
            raise ValueError("cosine_eta_min_ratio must satisfy 0.0 <= cosine_eta_min_ratio < 1.0.")
        eta_min = float(opt.param_groups[0]["lr"]) * cosine_eta_min_ratio
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_steps, eta_min=eta_min)
    elif lr_schedule == "linear":
        main_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=main_steps)
    else:
        raise ValueError(f"Unknown lr_schedule='{lr_schedule}'")

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps],
        )
    else:
        scheduler = main_scheduler

    # Checkpoint and lightweight diagnostics state.
    ckpt_path = Path(ckpt_dir) if ckpt_dir is not None else None
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    train_config = {
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "cosine_eta_min_ratio": cosine_eta_min_ratio,
        "weight_decay": weight_decay,
        "use_adamw": use_adamw,
        "grad_clip_norm": grad_clip_norm,
        "dtype": str(dtype),
        "device": str(device),
        "data_device": str(target.device),
        "pin_memory": pin,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "stats_freq_epochs": stats_freq_epochs,
        "log_grad_norm": log_grad_norm,
        "validation_samples": 0 if validation is None else len(validation),
        "restore_best": restore_best,
    }
    stats: dict[str, Any] = {"epoch": [], "training_loss": []}
    if validation is not None:
        stats["validation_loss"] = []
    if log_grad_norm:
        stats["grad_norm"] = []

    logger.info("train | epochs=%d bs=%d lr=%g schedule=%s opt=%s wd=%g device=%s",
                n_epochs, batch_size, lr, lr_schedule, "AdamW" if use_adamw else "Adam", weight_decay, device)

    # Main optimization loop.
    last_epoch_loss = float("nan")
    last_validation_loss = float("nan")
    best_epoch, best_validation_loss, best_state = 0, float("inf"), None
    for epoch in range(n_epochs):
        epoch_idx = epoch + 1
        epoch_loss_sum: torch.Tensor | None = None
        epoch_loss_count = 0

        perm = torch.randperm(len(target), device=target.device)
        for start in range(0, len(target), batch_size):
            x = target[perm[start: start + batch_size]]
            if target_pin:
                x = x.pin_memory()
            x = x.to(device=device, dtype=dtype, non_blocking=pin)

            z = None
            if source is not None:
                idx = torch.randint(0, source.size(0), (x.size(0),), device=source.device)
                z = source.index_select(0, idx)
                if source_pin:
                    z = z.pin_memory()
                z = z.to(device=device, dtype=dtype, non_blocking=source_pin)

            opt.zero_grad(set_to_none=True)
            loss = generative_model.loss(x, z=z)
            if loss.ndim != 0:
                raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            opt.step()
            scheduler.step()

            loss_value = loss.detach()
            epoch_loss_sum = loss_value if epoch_loss_sum is None else epoch_loss_sum + loss_value
            epoch_loss_count += 1

        # Epoch-end bookkeeping is deliberately light.
        should_record = epoch_idx % stats_freq_epochs == 0 or epoch_idx == n_epochs
        should_log = freq_logging > 0 and epoch_idx % freq_logging == 0
        should_checkpoint = ckpt_path is not None and epoch_idx % ckpt_freq_epochs == 0

        if should_record or should_log or should_checkpoint:
            if epoch_loss_sum is None or epoch_loss_count == 0:
                last_epoch_loss = float("nan")
            else:
                last_epoch_loss = float(epoch_loss_sum / epoch_loss_count)

        if validation is not None and (should_record or should_log):
            validation_loss = 0.0
            net.eval()
            rng_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                           if device.type == "cuda" else [])
            # Fixed diffusion noise makes changes in validation loss comparable across epochs.
            with torch.inference_mode(), torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(0)
                for start in range(0, len(validation), batch_size):
                    x = validation[start: start + batch_size]
                    if validation_pin:
                        x = x.pin_memory()
                    x = x.to(device=device, dtype=dtype, non_blocking=validation_pin)

                    z = None
                    if source is not None:
                        idx = torch.randint(0, source.size(0), (x.size(0),), device=source.device)
                        z = source.index_select(0, idx)
                        if source_pin:
                            z = z.pin_memory()
                        z = z.to(device=device, dtype=dtype, non_blocking=source_pin)

                    loss = generative_model.loss(x, z=z)
                    if loss.ndim != 0:
                        raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")
                    validation_loss += float(loss) * len(x)
            last_validation_loss = validation_loss / len(validation)
            if last_validation_loss < best_validation_loss:
                best_epoch, best_validation_loss = epoch_idx, last_validation_loss
                if restore_best:
                    best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
            net.train()

        if should_record:
            stats["epoch"].append(epoch_idx)
            stats["training_loss"].append(last_epoch_loss)
            if validation is not None:
                stats["validation_loss"].append(last_validation_loss)
            if log_grad_norm:
                grad_sq_sum = 0.0
                for param in net.parameters():
                    if param.grad is not None:
                        grad_sq_sum += float(param.grad.detach().pow(2).sum())
                grad_norm = math.sqrt(grad_sq_sum) if grad_sq_sum > 0.0 else None
                stats["grad_norm"].append(float("nan") if grad_norm is None else grad_norm)

        if should_log:
            lr_now = opt.param_groups[0]["lr"]
            validation_log = f" | validation {last_validation_loss:.6f}" if validation is not None else ""
            logger.info(f"epoch {epoch_idx:3d}/{n_epochs:3d} | loss {last_epoch_loss:.6f}{validation_log} | lr {lr_now:.3e}")

        if should_checkpoint:
            _save_ckpt(ckpt_path, ckpt_keep_last, epoch_idx, epoch_idx * steps_per_epoch,
                       last_epoch_loss, net, opt, scheduler, train_config)

    if validation is not None:
        stats.update({"best_epoch": best_epoch, "best_validation_loss": best_validation_loss})
    if best_state is not None:
        net.load_state_dict(best_state)
        logger.info("train | restored best validation weights from epoch %d", best_epoch)

    if ckpt_path is not None:
        _save_ckpt(ckpt_path, ckpt_keep_last, n_epochs, n_epochs * steps_per_epoch,
                   last_epoch_loss, net, opt, scheduler, train_config)

    logger.info("train | done")
    return generative_model, {"train_config": train_config, "stats": stats}


def _save_ckpt(ckpt_path, ckpt_keep_last, epoch_idx, global_step, last_loss, net, opt, scheduler, train_config) -> None:
    """Persist a checkpoint and rotate old ones."""
    if ckpt_path is None:
        return

    fname = ckpt_path / f"ckpt_epoch_{epoch_idx:04d}.pt"
    torch.save({
        "epoch": epoch_idx,
        "global_step": global_step,
        "loss": float(last_loss),
        "model_state": net.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_config": train_config,
    }, fname)

    last = ckpt_path / "ckpt_last.pt"
    try:
        if last.exists() or last.is_symlink():
            last.unlink()
        last.symlink_to(fname.name)
    except Exception:
        torch.save(torch.load(fname), last)

    if ckpt_keep_last > 0:
        ckpts = sorted(ckpt_path.glob("ckpt_epoch_*.pt"))
        for path in ckpts[: max(len(ckpts) - ckpt_keep_last, 0)]:
            try:
                path.unlink()
            except Exception:
                pass
