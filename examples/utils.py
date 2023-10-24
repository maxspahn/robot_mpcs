import yaml
import pybullet

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup

def visualize_constraints(fsd, height):
    plot_points = fsd.get_points()
    pybullet.removeAllUserDebugItems()
    for plot_point in plot_points:
        start_point = (plot_point[0, 0], plot_point[-1, 0], height)
        end_point = (plot_point[0, 1], plot_point[-1, 1], height)
        pybullet.addUserDebugLine(start_point, end_point)