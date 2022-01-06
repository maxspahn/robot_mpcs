# Robot mpcs

## Install forces pro
You have to request a license for [forcespro](https://forces.embotech.com/) and install it according to their
documentation.

## Install using poetry
Then you can install this package using [poetry](https://python-poetry.org/docs/) within a
virtual environment.
```bash
poetry install
poetry shell
```
Now you are in the virtual environment with everything installed.

## Install globally using pip
```bash
pip install .
```

## Examples
```
cd examples
python3 makeSolver.py
python3 mpcPlanner.py
```

This is a minimal example using a point robot.
