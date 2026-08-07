"""Visualize 2D sampling results and trajectories."""

import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from gendynamics import DDPMV, GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOTLinear, FlowMatchingOrigin, ScoreSDEOrigin
from gendynamics.datasets import fetch_synthetic_data
from gendynamics.metrics import sliced_wasserstein
from gendynamics.nn import MLPModel
from gendynamics.training import train


####################################################################################################
# Main
if __name__ == "__main__":

    t0 = time.time()

    figures_dir = "_figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--blank", action="store_true", default=False, help="CI helper.")
    parser.add_argument("--model", type=str, default='gf_linear', help="Model.")
    parser.add_argument("--data", type=str, default='checker', help="Data distribution.")
    args = parser.parse_args()

    dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 100_000 if not args.blank else 100
    n_test_samples = 5000
    n_visu_samples = 25
    n_steps = 512
    batch_size = 1024
    n_epochs = 256
    lr = 5e-4
    width = 256
    depth = 5
    fdtype = torch.float32
    idtype = torch.int32
    num_workers = 0

    if args.model == "ddpm":
        model = DDPMV
    elif args.model == "gf_linear":
        model = GaussianFlowLinear
    elif args.model == "gf_ot_linear":
        model = GaussianFlowOTLinear
    elif args.model == "gf_ddpm":
        model = GaussianFlowDDPM
    elif args.model == "fm_origin":
        model = FlowMatchingOrigin
    elif args.model == "sde_origin":
        model = ScoreSDEOrigin
    else:
        raise ValueError(f"Unknown model {args.model!r}.")

    print(f"[INFO] 2D sample visualization | dataset={args.data} | device={device} | n_samples={n_samples}"
          f" | n_steps={n_steps} | batch_size={batch_size} | n_epochs={n_epochs}")

    x_train, _, x_ref = fetch_synthetic_data(args.data, n_samples=n_samples, dim=dim, device=device, dtype=fdtype)
    x_ref = x_ref[:n_test_samples]
    train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, num_workers=num_workers)

    print(f"[INFO] Train {model.__name__} | dataset={args.data}"
          f" | epochs={n_epochs} | batch_size={batch_size} | lr={lr:.1e}")

    net = MLPModel(input_dim=dim, width=width, depth=depth).to(device=device, dtype=fdtype)
    generator = model(net=net, dim=dim, fdtype=fdtype, idtype=idtype, device=device, n_steps=n_steps)
    train(generative_model=generator, **train_kwargs)
    x_gen = generator.sample(n_samples=n_test_samples)

    print(f"[INFO] Eval {model.__name__} | Wasserstein-dist={sliced_wasserstein(x_ref, x_gen):.2e}")

    x_np = x_gen.detach().cpu().numpy()
    x_ref_np = x_ref.detach().cpu().numpy()
    center = np.median(np.vstack([x_np, x_ref_np]), axis=0)
    quant_x, quant_y = np.quantile(np.abs(np.vstack([x_np, x_ref_np]) - center), 0.95, axis=0)

####################################################################################################
# Plotting
    plt.figure(figsize=(5, 4))

    plt.scatter(x_ref_np[:, 0], x_ref_np[:, 1], s=6, label="Target", alpha=0.6)
    plt.scatter(x_np[:, 0], x_np[:, 1], s=6, label="Generated", alpha=0.6)
    plt.xlim(center[0] - 1.05 * quant_x, center[0] + 1.05 * quant_x)
    plt.ylim(center[1] - 1.05 * quant_y, center[1] + 1.05 * quant_y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=10)

    plt.tight_layout(pad=0.8)

    filepath = figures_dir / f"{args.data}_{model.__name__}_2d_scatter.pdf"
    plt.savefig(filepath, dpi=300)

    if hasattr(generator, "_sample"):
        _, trajectory = generator._sample(n_visu_samples, return_trajectory=True)
        trajectory_np = torch.stack(trajectory).detach().cpu().numpy()
        trajectory_points = trajectory_np.reshape(-1, dim)
        lo = trajectory_points.min(axis=0)
        hi = trajectory_points.max(axis=0)
        pad = np.maximum(0.05 * (hi - lo), 1e-3)

        plt.figure(figsize=(5, 4))
        for i in range(trajectory_np.shape[1]):
            path = trajectory_np[:, i, :]
            plt.plot(path[:, 0], path[:, 1], lw=1.2, color="gray", alpha=0.5)
            plt.scatter(path[0, 0], path[0, 1], marker="x", s=28, color="gray", alpha=0.5, label="Start" if i == 0 else None)
            plt.scatter(path[-1, 0], path[-1, 1], marker="o", s=20, color="gray", alpha=0.5, label="Final" if i == 0 else None)
        plt.xlim(lo[0] - pad[0], hi[0] + pad[0])
        plt.ylim(lo[1] - pad[1], hi[1] + pad[1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(fontsize=10)
        plt.tight_layout(pad=0.8)

        trajectory_filepath = figures_dir / f"{args.data}_{model.__name__}_2d_trajectories.pdf"
        plt.savefig(trajectory_filepath, dpi=300)
        print(f"[INFO] saved={trajectory_filepath}")
    else:
        print(f"[INFO] trajectory plot skipped: {model.__name__} does not expose intermediate sampling states")

    print(f"[INFO] saved={filepath} | total={time.time() - t0:.1f}s")
