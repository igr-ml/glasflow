# Contributing to glasflow

## Installation

To install `glasflow` and contribute clone the repo and install the additional dependencies with:

```console
$ cd glasflow
$ pip install -e .[dev]
```

## Format checking

We use [pre-commit](https://pre-commit.com/) to re-format code using `black` and check the quality of code suing `flake8` before committing.

This requires some setup:

```console
$ pip install pre-commit # Should already be installed
$ cd glasflow
$ pre-commit install
```

Now we you run `$ git commit` `pre-commit` will run a series of checks. Some checks will automatically change the code and others will print warnings that you must address and re-commit.

## Testing glasflow

When contributing code to `glasflow` please ensure that you also contribute corresponding unit tests and integration tests where applicable. We test `glasflow` using `pytest` and strive to test all of the core functionality in `glasflow`. Tests should be contained with the `tests` directory and follow the naming convention `test_<name>.py`. We also welcome improvements to the existing tests and testing infrastructure.

The tests can be run from the root directory using

```console
$ pytest
```

Specific tests can be run using

```console
$ pytest tests/test_<name>.py
```

See the `pytest` [documentation](https://docs.pytest.org/) for further details on how to write tests using `pytest`.
