# Robot MPC Ros Wrapper

Simple ros-noetic wrapper for boxer robot by clearpath.


## Installation

Navigate to root directory of `robot_mpcs`.
Every command assumes you are in the root directory of `robot_mpcs`.

Clone the boxer simulation environment using the provided submodules.

```bash
git submodule update --init --recursive
git submodule update --recursive --remote
```

Install `robot_mpcs` globally (or in the virtual environment if you use ros inside one).
```bash
pip install -e .
```
ira_laser_tools

Build catkin workspace
```bash
cd ros_bridge
catkin build
```

Generate the solver.
```bash
cd ros_bridge
source devel/setup.{zsh|bash}
roscd robotmpcs_ros/script
python3 make_solver.py ../config/boxer_mpc_config.yaml
```

Launch mpc planner
```bash
cd ros_bridge
source devel/setup.{zsh|bash}
roslaunch boxer_gazebo boxer_world.launch
roslaunch robotmpcs_ros boxer_mpc.launch
```

Now, you can see a topic to publish the goal to.

```bash
rostopic pub /mpc/goal std_msgs/Float64MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data: [0.0, 1.0]"
```


