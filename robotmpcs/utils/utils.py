import casadi as ca
import numpy as np
import yaml

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup

def visualize_constraints_in_pybullet(fsd, height):
    import pybullet
    plot_points = fsd.get_points()
    pybullet.removeAllUserDebugItems()
    for plot_point in plot_points:
        start_point = (plot_point[0, 0], plot_point[-1, 0], height)
        end_point = (plot_point[0, 1], plot_point[-1, 1], height)
        pybullet.addUserDebugLine(start_point, end_point)

def visualize_constraints_over_N_in_pybullet(halfplanes, height):
    import pybullet
    plot_points = []
    if len(halfplanes)>0:
        for constraint in halfplanes:
            if len(constraint) >0:
                plot_points.append(constraint[0].get_points())
        pybullet.removeAllUserDebugItems()
        for plot_point in plot_points:
            start_point = (plot_point[0, 0], plot_point[-1, 0], height)
            end_point = (plot_point[0, 1], plot_point[-1, 1], height)
            pybullet.addUserDebugLine(start_point, end_point)

def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a

def extractVariables(z, n, nx, nu, ns):
    q = z[0: n]
    qdot = z[n: nx]
    qddot = z[nx + ns : nx + ns + nu]
    return q, qdot, qddot

def get_velocity(z, n, nx):
    return  z[n: nx]


def point_to_plane(point: ca.SX, plane: ca.SX) -> ca.SX:
    distance = ca.fabs(ca.dot(plane[0:3], point) + plane[3]) / ca.norm_2(
        plane[0:3]
    )
    return distance

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup
