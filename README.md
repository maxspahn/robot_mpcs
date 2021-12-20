# Robot mpcs

This is a very basic implementation of several mpc planners for simple robots.
The actuation is always the accelerations of individual joints.

## Installation
```bash
git clone git@github.com:maxspahn/robot_mpcs.git
pip3 install -e .
```

Now you can use the package.

## Examples
```
cd examples
python3 makeSolver.py
python3 mpcPlanner.py
```

This is a minimal example using a point robot.
