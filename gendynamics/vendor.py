"""Install the PhysicsNeMo source required by the t-EDM adapter."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from .thirdparty import _import_tedm_vendor


def ensure_tedm_vendor() -> Path:
    """Fetch PhysicsNeMo once into the installed gendynamics package."""
    root = Path(__file__).resolve().parent / "_vendor"
    target = root / "physicsnemo"
    if (target / "physicsnemo" / "diffusion").is_dir():
        _import_tedm_vendor(str(target))
        return target

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        staged = Path(temporary) / "physicsnemo"
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2.0.0",
                        "https://github.com/NVIDIA/physicsnemo.git", str(staged)],
                       check=True)
        if not (staged / "physicsnemo" / "diffusion").is_dir():
            raise FileNotFoundError("The downloaded PhysicsNeMo source has no diffusion package")
        if target.exists():
            raise FileExistsError(f"Incomplete PhysicsNeMo vendor directory: {target}")
        shutil.move(str(staged), target)
    _import_tedm_vendor(str(target))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", choices=("tedm",))
    parser.parse_args()
    print(ensure_tedm_vendor())
