# Robot mpcs

## Install forces pro
You have to request a license for [forcespro](https://forces.embotech.com/) and install it
according to their documentation.
The location of the python package `forcespro` must also be included in your python path.
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/forces/pro"
```
Consider adding it to your `.bashrc` (`.zshrc`) file

## Install using poetry
Then you can install this package using [poetry](https://python-poetry.org/docs/) within a
virtual environment.
```bash
poetry install -E agents
poetry shell
```
Now you are in the virtual environment with everything installed.

## Install globally using pip
```bash
pip3 install .
```
If you want to test the mpc solvers you need to install additional dependencies.
```bash
pip3 install '.[agents]'
```

## Examples
```
cd examples
python3 makeSolver.py <path/to/config/file>
python3 <robot/type>_example.py <path/to/config/file>
```

