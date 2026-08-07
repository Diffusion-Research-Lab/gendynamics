# GenDynamics

GenDynamics provides compact PyTorch implementations of diffusion and flow-matching models, together with training, sampling, evaluation, and synthetic-data utilities.

## Installation

```bash
pip install "gendynamics @ git+ssh://git@github.com/Diffusion-Research/gendynamics.git"
```

For development:

```bash
git clone git@github.com:Diffusion-Research/gendynamics.git
cd gendynamics
python -m pip install -e ".[dev,examples]"
```

Optional reference implementations can be fetched into the ignored `gendynamics/_vendor` directory:

```bash
bash scripts/fetch.vendor.sh
```

## Minimal example

```python
from gendynamics.datasets import fetch_synthetic_data
from gendynamics.flow_matching import GaussianFlowLinear
from gendynamics.nn import MLPModel
from gendynamics.training import train

x_train, _, _ = fetch_synthetic_data("checker", n_samples=10_000)
net = MLPModel(input_dim=2, width=64, depth=4, time_dim=32)
generator = GaussianFlowLinear(net=net, dim=2, n_steps=128)
generator, _ = train(generator, target_data=x_train, n_epochs=128, batch_size=1024, lr=1e-3)
x_gen = generator.sample(n_samples=2048)
```

## Development

```bash
make setup
make vendor
make check
```

## License

GenDynamics is released under the MIT License.
