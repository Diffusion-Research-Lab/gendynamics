# GenDynamics

PyTorch diffusion and flow-matching models with training, sampling, metrics, and synthetic datasets.

```bash
pip install "gendynamics @ git+https://github.com/Diffusion-Research-Lab/gendynamics.git"
```

For local development:

```bash
git clone git@github.com:Diffusion-Research-Lab/gendynamics.git
cd gendynamics
python -m pip install -e ".[dev,examples]"
```

Fetch optional reference implementations with:

```bash
bash scripts/fetch.vendor.sh
```

Run the local checks with:

```bash
make setup
make check
```
