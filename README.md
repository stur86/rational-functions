# rational-functions

A Numpy-based Python package for the manipulation of rational functions.

## Installation

Install with:

```bash
pip install rational-functions
```

## Usage

Import the main `RationalFunction` class directly from the package:

```py
from rational_functions import RationalFunction
```

For more information see [the documentation](https://stur86.github.io/rational-functions/).

## Development

Contributions are welcome! Please fork the repository and send a PR for any suggested improvements or fixes.

The package uses UV as its package manager, [refer to their documentation](https://docs.astral.sh/uv/) for details about usage.

After having the environment set up I recommend using the pre-commit hooks:

```bash
pre-commit install
```

You can also use Ruff and Black to clean up your code by using `uv run task lint` and `uv run task format`.
