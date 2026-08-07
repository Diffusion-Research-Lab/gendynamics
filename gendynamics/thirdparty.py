"""Adapters around vendored third-party generative backends."""

import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import types
import torch
import torch.nn.functional as F
from ._abs import Base
from ._noise import sample_gaussian

__all__ = [
    "ScoreSDEOrigin",
    "FlowMatchingOrigin",
    "DLPMEpsOrigin",
    "TEDMOrigin",
]


def _resolve_fdtype(fdtype: Optional[torch.dtype], dtype: Optional[torch.dtype]) -> torch.dtype:
    if fdtype is None and dtype is None:
        return torch.float32
    if dtype is None:
        return torch.float32 if fdtype is None else fdtype
    if fdtype is not None and fdtype != dtype:
        raise ValueError(f"Conflicting fdtype={fdtype} and legacy dtype={dtype}.")
    warnings.warn(
        "`dtype` is deprecated in thirdparty adapters and will be removed in v0.2; use `fdtype` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if fdtype is None:
        return dtype
    return fdtype


def _vendor_root(package_root: Optional[str], vendor_name: str) -> Path:
    if package_root is not None:
        return Path(package_root).expanduser().resolve()
    return Path(__file__).resolve().parent / "_vendor" / vendor_name


def _load_vendor_module(module_name: str, file_path: Path) -> types.ModuleType:
    """Load one vendored module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name!r} from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_package_stub(package_name: str, package_path: Path, saved_modules: dict[str, object]) -> None:
    """Install a minimal package stub so file-loaded modules can resolve relatives."""
    if package_name not in saved_modules:
        saved_modules[package_name] = sys.modules.get(package_name, None)
    stub = types.ModuleType(package_name)
    stub.__path__ = [str(package_path)]
    sys.modules[package_name] = stub


def _install_module_stub(module_name: str, module: types.ModuleType, saved_modules: dict[str, object]) -> None:
    """Install a temporary module stub and remember any previous module."""
    if module_name not in saved_modules:
        saved_modules[module_name] = sys.modules.get(module_name, None)
    sys.modules[module_name] = module


def _import_vendor(
    package_root: Optional[str],
    vendor_name: str,
    modules: list[tuple[str, tuple[str, ...]]],
    cleanup_prefixes: tuple[str, ...] = (),
) -> tuple:
    root = _vendor_root(package_root, vendor_name)
    modules_before = set(sys.modules) if cleanup_prefixes else None
    prev_warp_quiet = os.environ.get("WARP_QUIET")
    os.environ["WARP_QUIET"] = "1"
    sys.path.insert(0, str(root))
    try:
        values = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for module_name, names in modules:
                module = importlib.import_module(module_name)
                values.extend(getattr(module, name) for name in names)
        return tuple(values)
    finally:
        sys.path[:] = [entry for entry in sys.path if entry != str(root)]
        if prev_warp_quiet is None:
            os.environ.pop("WARP_QUIET", None)
        else:
            os.environ["WARP_QUIET"] = prev_warp_quiet
        if modules_before is not None:
            loaded_modules = set(sys.modules) - modules_before
            for name in loaded_modules:
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in cleanup_prefixes):
                    sys.modules.pop(name, None)


def _import_tedm_vendor(package_root: Optional[str]):
    """Import TEDM components without running the vendor package initializers."""
    root = _vendor_root(package_root, "physicsnemo")
    package_dir = root / "physicsnemo" if (root / "physicsnemo").exists() else root
    modules_before = set(sys.modules)
    saved_modules: dict[str, object] = {}

    def _first_existing(*paths: Path) -> Path:
        for path in paths:
            if path.exists():
                return path
        raise FileNotFoundError(f"None of the TEDM vendor files exist under {root}.")

    try:
        _install_package_stub("physicsnemo", package_dir, saved_modules)
        _install_package_stub("physicsnemo.diffusion", package_dir / "diffusion", saved_modules)
        _install_package_stub("physicsnemo.core", package_dir / "core", saved_modules)
        _install_package_stub("physicsnemo.domain_parallel", package_dir / "domain_parallel", saved_modules)

        core_meta_stub = types.ModuleType("physicsnemo.core.meta")

        class _ModelMetaData:
            pass

        core_meta_stub.ModelMetaData = _ModelMetaData
        _install_module_stub("physicsnemo.core.meta", core_meta_stub, saved_modules)

        core_module_stub = types.ModuleType("physicsnemo.core.module")

        class _Module(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        core_module_stub.Module = _Module
        _install_module_stub("physicsnemo.core.module", core_module_stub, saved_modules)

        shard_tensor_stub = types.ModuleType("physicsnemo.domain_parallel.shard_tensor")

        def _scatter_tensor(tensor, *args, **kwargs):
            return tensor

        shard_tensor_stub.scatter_tensor = _scatter_tensor
        _install_module_stub("physicsnemo.domain_parallel.shard_tensor", shard_tensor_stub, saved_modules)

        noise_path = _first_existing(
            package_dir / "diffusion" / "noise_schedulers" / "noise_schedulers.py",
            package_dir / "diffusion" / "noise_schedulers.py",
        )
        noise_module = _load_vendor_module("physicsnemo.diffusion.noise_schedulers", noise_path)

        preconditioner_path = _first_existing(
            package_dir / "diffusion" / "preconditioners" / "preconditioners.py",
            package_dir / "diffusion" / "preconditioners.py",
        )
        preconditioner_module = _load_vendor_module("physicsnemo.diffusion.preconditioners", preconditioner_path)

        sampler_path = _first_existing(
            package_dir / "diffusion" / "samplers" / "samplers.py",
            package_dir / "diffusion" / "samplers.py",
        )
        _install_package_stub("physicsnemo.diffusion.samplers", sampler_path.parent, saved_modules)
        sampler_module = _load_vendor_module("physicsnemo.diffusion.samplers.samplers", sampler_path)

        return (
            noise_module.StudentTEDMNoiseScheduler,
            preconditioner_module.EDMPreconditioner,
            sampler_module.sample,
        )
    finally:
        loaded_modules = set(sys.modules) - modules_before
        for name in loaded_modules:
            if name == "physicsnemo" or name.startswith("physicsnemo."):
                sys.modules.pop(name, None)
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _net_device_dtype(model: torch.nn.Module, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters(), None)
    if param is None:
        return fallback.device, fallback.dtype
    return param.device, param.dtype


def _as_time_column(t, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.expand(batch_size)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    return t.to(device=device, dtype=dtype)


class _VectorNet(torch.nn.Module):
    """Adapt a gendynamics vector net to vendor `(x, t)` conventions."""

    def __init__(self, model: torch.nn.Module, *, with_channel: bool = False):
        super().__init__()
        self.model = model
        self.with_channel = bool(with_channel)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        device, dtype = _net_device_dtype(self.model, x)
        x = x.to(device=device, dtype=dtype)
        if self.with_channel:
            x = x.squeeze(1)
        t = _as_time_column(t, x.size(0), device=device, dtype=dtype)
        out = self.model(x, t)
        return out.unsqueeze(1) if self.with_channel else out


class _ScoreNet(torch.nn.Module):
    """Adapt a vector net to the `(B, D, 1, 1)` score-SDE interface."""

    def __init__(self, model: torch.nn.Module, dim: int):
        super().__init__()
        self.model = _VectorNet(model)
        self.dim = int(dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), self.dim)
        return self.model(x, labels).reshape(-1, self.dim, 1, 1)


class DLPMEpsOrigin(Base):
    _family = "vendor"
    _loss_tag = "sqrt_mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        alpha: float = 1.8,
        authors_root: Optional[str] = None,
        time_spacing: str = "linear",
        rescale_timesteps: bool = True,
        isotropic: bool = True,
        loss_monte_carlo: str = "mean",
        monte_carlo_outer: int = 1,
        monte_carlo_inner: int = 1,
        lploss: float = 2.0,
        clamp_a: Optional[float] = None,
        clamp_eps: Optional[float] = None,
        scale: str = "scale_preserving",
        base_or_sample: torch.Tensor = None,
        fdtype: Optional[torch.dtype] = None,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(
            net=net,
            dim=dim,
            n_steps=n_steps,
            base_or_sample=base_or_sample,
            fdtype=fdtype,
            idtype=idtype,
            device=device,
        )
        if base_or_sample is not None:
            warnings.warn("DLPMEpsOrigin ignores base_or_sample during sampling.")

        (GenerativeLevyProcess,) = _import_vendor(
            authors_root,
            "DLPM",
            [("dlpm.methods.GenerativeLevyProcess", ("GenerativeLevyProcess",))],
            cleanup_prefixes=("dlpm",),
        )

        self._vendor_net = _VectorNet(self._net, with_channel=True)
        self._loss_monte_carlo = loss_monte_carlo
        self._monte_carlo_outer = monte_carlo_outer
        self._monte_carlo_inner = monte_carlo_inner
        self._lploss = lploss
        self._clamp_a = clamp_a
        self._clamp_eps = clamp_eps
        self._glp = GenerativeLevyProcess(
            alpha=float(alpha),
            device=self._device,
            reverse_steps=int(n_steps),
            time_spacing=str(time_spacing),
            rescale_timesteps=bool(rescale_timesteps),
            isotropic=bool(isotropic),
            scale=str(scale),
        )

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        raise ValueError("In DLPMEpsOrigin sampling is handled internally by the DLPM sampler.")

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if z is not None:
            warnings.warn("In 'DLPMEpsOrigin.loss', input 'z' is ignored (z = A G are sampled internally).")

        x = x.to(device=self._device, dtype=self._fdtype).unsqueeze(1)
        out = self._glp.training_losses(
            models={"default": self._vendor_net},
            x_start=x,
            loss_type="EPS_LOSS",
            lploss=self._lploss,
            loss_monte_carlo=self._loss_monte_carlo,
            monte_carlo_outer=self._monte_carlo_outer,
            monte_carlo_inner=self._monte_carlo_inner,
            clamp_a=self._clamp_a,
            clamp_eps=self._clamp_eps,
        )
        return out["loss"].to(device=self._device, dtype=self._fdtype).mean()

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        self._net.eval()
        if self._clamp_a is not None:
            self._glp.dlpm.gen_a.setParams(clamp_a=self._clamp_a)
        if self._clamp_eps is not None:
            self._glp.dlpm.gen_eps.setParams(clamp_eps=self._clamp_eps)
        x = self._glp.p_sample_loop(
            model=self._vendor_net,
            shape=(int(n_samples), 1, int(self._dim)),
            progress=bool(kwargs.get("progress", False)),
        )
        return x.squeeze(1).to(device=self._device, dtype=self._fdtype)


class FlowMatchingOrigin(Base):
    _family = "vendor"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        t_min: float = 0.0,
        t_max: float = 1.0,
        package_root: Optional[str] = None,
        base_or_sample: torch.Tensor = None,
        fdtype: Optional[torch.dtype] = None,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(
            net=net,
            dim=dim,
            n_steps=n_steps,
            base_or_sample=base_or_sample,
            fdtype=fdtype,
            idtype=idtype,
            device=device,
        )
        if float(t_min) != 0.0 or float(t_max) != 1.0:
            warnings.warn("FlowMatchingOrigin always uses t_min=0 and t_max=1; passed values are ignored.")

        AffineProbPath, CondOTScheduler, ODESolver = _import_vendor(
            package_root,
            "flow_matching",
            [
                ("flow_matching.path", ("AffineProbPath",)),
                ("flow_matching.path.scheduler", ("CondOTScheduler",)),
                ("flow_matching.solver", ("ODESolver",)),
            ],
            cleanup_prefixes=("flow_matching",),
        )

        self._t_min = 0.0
        self._t_max = 1.0
        self._path = AffineProbPath(scheduler=CondOTScheduler())
        self._solver = ODESolver(velocity_model=_VectorNet(self._net))

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        return sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._fdtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        x_0 = self._sample_source(x_1.size(0)) if z is None else z.to(device=self._device, dtype=self._fdtype)
        if x_0.shape != x_1.shape:
            raise ValueError(f"z must have shape {tuple(x_1.shape)}, got {tuple(x_0.shape)}.")

        t = torch.rand((x_1.size(0),), device=self._device, dtype=self._fdtype)
        path_sample = self._path.sample(x_0=x_0, x_1=x_1, t=t)
        v_hat = self._net(path_sample.x_t, path_sample.t.unsqueeze(-1))
        if v_hat.shape != path_sample.dx_t.shape:
            raise ValueError(f"Shape mismatch: v_hat={tuple(v_hat.shape)} vs dx_t={tuple(path_sample.dx_t.shape)}")
        return F.mse_loss(v_hat, path_sample.dx_t)

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        self._net.eval()
        x_init = self._sample_source(int(n_samples))
        time_grid = torch.tensor([self._t_min, self._t_max], device=self._device, dtype=self._fdtype)
        step_size = (self._t_max - self._t_min) / float(max(self._n_steps, 1))
        x = self._solver.sample(
            x_init=x_init,
            time_grid=time_grid,
            method="midpoint",
            step_size=step_size,
            return_intermediates=False,
        )
        return x.to(device=self._device, dtype=self._fdtype)


class ScoreSDEOrigin(Base):
    _family = "vendor"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        loss_eps: float = 1e-5,
        sampling_eps: float = 1e-5,
        reduce_mean: bool = False,
        time_eps: Optional[float] = None,
        use_corrector: Optional[bool] = None,
        snr: Optional[float] = None,
        corrector_steps: int = 1,
        package_root: Optional[str] = None,
        base_or_sample: torch.Tensor = None,
        fdtype: Optional[torch.dtype] = None,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(
            net=net,
            dim=dim,
            n_steps=n_steps,
            base_or_sample=base_or_sample,
            fdtype=fdtype,
            idtype=idtype,
            device=device,
        )
        if base_or_sample is not None:
            warnings.warn("ScoreSDEOrigin ignores base_or_sample during sampling.")

        if time_eps is not None:
            warnings.warn("`time_eps` is deprecated in ScoreSDEOrigin; use `loss_eps` and `sampling_eps` instead.")
            loss_eps = float(time_eps)
            sampling_eps = float(time_eps)
        if not (0.0 < float(loss_eps) < 1.0):
            raise ValueError(f"loss_eps must be in (0,1), got {loss_eps}.")
        if not (0.0 < float(sampling_eps) < 1.0):
            raise ValueError(f"sampling_eps must be in (0,1), got {sampling_eps}.")

        root = _vendor_root(package_root, "score_sde_pytorch")
        modules_before = set(sys.modules)
        sys.path.insert(0, str(root))
        try:
            sde_lib = importlib.import_module("sde_lib")
            losses = importlib.import_module("losses")
            sampling = importlib.import_module("sampling")
            VESDE = sde_lib.VESDE
            get_sde_loss_fn = losses.get_sde_loss_fn
            ReverseDiffusionPredictor = sampling.ReverseDiffusionPredictor
            NoneCorrector = sampling.NoneCorrector
            get_pc_sampler = sampling.get_pc_sampler
            LangevinCorrector = getattr(sampling, "LangevinCorrector", None)
        finally:
            sys.path[:] = [entry for entry in sys.path if entry != str(root)]
            loaded_modules = set(sys.modules) - modules_before
            for name in loaded_modules:
                if name in {"sde_lib", "losses", "sampling", "models", "utils"} or any(
                    name.startswith(f"{prefix}.") for prefix in ("models", "utils")
                ):
                    sys.modules.pop(name, None)

        # Image-scale VE defaults over-noise tiny vector problems.
        low_dim = int(dim) <= 4
        if sigma_min is None:
            sigma_min = 1e-2
        if sigma_max is None:
            sigma_max = 2.0 if low_dim else 50.0

        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._loss_eps = float(loss_eps)
        self._sampling_eps = float(sampling_eps)
        if use_corrector is None:
            use_corrector = (not low_dim) and LangevinCorrector is not None
        self._snr = 0.16 if snr is None and use_corrector else 0.0 if snr is None else float(snr)
        self._corrector_steps = int(corrector_steps)
        self._sde = VESDE(sigma_min=self._sigma_min, sigma_max=self._sigma_max, N=int(n_steps))
        self._loss_fn = get_sde_loss_fn(
            self._sde,
            train=True,
            reduce_mean=bool(reduce_mean),
            continuous=True,
            likelihood_weighting=False,
            eps=self._loss_eps,
        )
        self._get_pc_sampler = get_pc_sampler
        self._predictor_cls = ReverseDiffusionPredictor
        self._corrector_cls = LangevinCorrector if use_corrector and LangevinCorrector is not None else NoneCorrector
        self._score_model = _ScoreNet(self._net, self._dim)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        return self._sigma_max * sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._fdtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if z is not None:
            warnings.warn("In 'ScoreSDEOrigin.loss', input 'z' is ignored (Gaussian perturbations are sampled internally).")
        x = x.to(device=self._device, dtype=self._fdtype).reshape(-1, self._dim, 1, 1)
        return self._loss_fn(self._score_model, x)

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        self._net.eval()
        sampling_fn = self._get_pc_sampler(
            self._sde,
            (int(n_samples), int(self._dim), 1, 1),
            self._predictor_cls,
            self._corrector_cls,
            inverse_scaler=lambda x: x,
            snr=self._snr,
            n_steps=self._corrector_steps,
            probability_flow=False,
            continuous=True,
            denoise=True,
            eps=self._sampling_eps,
            device=self._device,
        )
        x, _ = sampling_fn(self._score_model)
        return x.reshape(int(n_samples), int(self._dim)).to(device=self._device, dtype=self._fdtype)


class TEDMOrigin(Base):
    _family = "vendor"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 18,
        nu: int = 10,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        solver: str = "edm_stochastic_heun",
        package_root: Optional[str] = None,
        base_or_sample: Optional[torch.Tensor] = None,
        fdtype: Optional[torch.dtype] = None,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(
            net=net,
            dim=dim,
            n_steps=n_steps,
            base_or_sample=base_or_sample,
            fdtype=fdtype,
            idtype=idtype,
            device=device,
        )
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
        if float(sigma_min) <= 0.0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}.")
        if float(sigma_max) <= float(sigma_min):
            raise ValueError(f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}.")
        if float(rho) <= 0.0:
            raise ValueError(f"rho must be positive, got {rho}.")
        if float(sigma_data) <= 0.0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}.")
        if float(p_std) <= 0.0:
            raise ValueError(f"p_std must be positive, got {p_std}.")

        self._nu = float(nu)
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._rho = float(rho)
        self._sigma_data = float(sigma_data)
        self._p_mean = float(p_mean)
        self._p_std = float(p_std)
        self._solver = str(solver)

        StudentTEDMNoiseScheduler, EDMPreconditioner, pn_sample = _import_tedm_vendor(package_root)
        self._pn_sample = pn_sample
        self._scheduler = StudentTEDMNoiseScheduler(
            sigma_min=self._sigma_min,
            sigma_max=self._sigma_max,
            rho=self._rho,
            nu=self._nu,
            sigma_data=self._sigma_data,
            P_mean=self._p_mean,
            P_std=self._p_std,
        )
        self._model = EDMPreconditioner(_VectorNet(self._net), sigma_data=self._sigma_data).to(
            device=self._device,
            dtype=self._fdtype,
        )

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        df = torch.tensor(float(self._scheduler.nu), device=self._device, dtype=self._fdtype)
        return torch.distributions.StudentT(df=df).rsample((n_samples, *self._sample_shape)).to(device=self._device, dtype=self._fdtype)

    def _resolve_time(self, t, n_samples: int) -> torch.Tensor:
        if t is None:
            return self._scheduler.sample_time(n_samples, device=self._device, dtype=self._fdtype)
        if isinstance(t, bool):
            raise TypeError("'t' must be a scalar or tensor, not bool.")
        if isinstance(t, (int, float)):
            return torch.full((n_samples,), float(t), device=self._device, dtype=self._fdtype)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Unsupported type for 't': {type(t).__name__}.")
        t = t.to(device=self._device, dtype=self._fdtype)
        if t.ndim == 0:
            return self._resolve_time(t.item(), n_samples)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1 or t.numel() != n_samples:
            raise ValueError(f"t must have shape ({n_samples},) or ({n_samples}, 1), got {tuple(t.shape)}.")
        return t

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None, **kwargs) -> torch.Tensor:
        x = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x.shape[1:]) != self._sample_shape:
            raise ValueError(f"x must have shape (B, *{self._sample_shape}), got {tuple(x.shape)}.")

        t = self._resolve_time(t, x.size(0))
        if z is None:
            x_t = self._scheduler.add_noise(x, t)
        else:
            z = z.to(device=self._device, dtype=self._fdtype)
            if z.shape != x.shape:
                raise ValueError(f"z must have shape {tuple(x.shape)}, got {tuple(z.shape)}.")
            x_t = x + self._expand_batch_scalar(self._scheduler.sigma(t), x) * z

        x_hat = self._model(x_t, t)
        if x_hat.shape != x.shape:
            raise ValueError(f"Shape mismatch: x_hat={tuple(x_hat.shape)} vs x={tuple(x.shape)}")

        w = self._scheduler.loss_weight(t)
        if w.ndim == 1:
            w = self._expand_batch_scalar(w, x)
        elif w.ndim == 2 and x.ndim >= 3 and w.shape == (x.size(0), x.size(1)):
            w = w.reshape(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
        else:
            raise ValueError(f"loss_weight returned shape {tuple(w.shape)}, incompatible with x shape {tuple(x.shape)}.")
        return (w * F.mse_loss(x_hat, x, reduction="none")).mean()

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        self._net.eval()
        t_steps = self._scheduler.timesteps(self._n_steps, device=self._device, dtype=self._fdtype)
        tN = t_steps[0].expand(n_samples)
        if self._base_or_sample is None:
            x_init = self._scheduler.init_latents(self._sample_shape, tN, device=self._device, dtype=self._fdtype)
        else:
            source = self._sample_source(n_samples)
            x_init = self._expand_batch_scalar(self._scheduler.sigma(tN), source) * source
        denoiser = self._scheduler.get_denoiser(x0_predictor=self._model)
        x = self._pn_sample(denoiser, x_init, self._scheduler, num_steps=self._n_steps, solver=self._solver)
        return x.to(device=self._device, dtype=self._fdtype)
