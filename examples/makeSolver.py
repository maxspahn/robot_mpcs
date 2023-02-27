import os
import sys
import re
import yaml
from robotmpcs.models.pandaMpcModel import PandaMpcModel

from robotmpcs.models.pointRobotMpcModel import PointRobotMpcModel
from robotmpcs.models.planarArmMpcModel import PlanarArmMpcModel
from robotmpcs.models.diffDriveMpcModel import DiffDriveMpcModel
from robotmpcs.models.boxerMpcModel import BoxerMpcModel

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup


def main(robot_type, setup_file):
    setup = parse_setup(setup_file)
    N = setup['H']
    m = setup['m']
    dt = setup['dt']
    number_obstacles = setup['obst']['nbObst']
    if robot_type == 'panda':
        mpcModel = PandaMpcModel(m, N)
    elif robot_type == 'boxer':
        mpcModel = BoxerMpcModel(N)
    elif robot_type == 'po1ntRobot':
        mpcModel = PointRobotMpcModel(m, N)
    else:
        print(f"{robot_type} is not a valid robot_type.")
        return
    if 'slack' in list(setup.keys()) and setup['slack'] == True:
        mpcModel.setSlack()
    mpcModel.setDt(dt)
    mpcModel.setObstacles(number_obstacles, m, inCostFunction=False)
    mpcModel.setModel()
    mpcModel.setCodeoptions()
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpcModel.generateSolver(location=path_to_solvers)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a config file for solver generation.")
    else:
        robot_type = re.findall('\/(\S*)M', sys.argv[1])[0]
        main(robot_type, sys.argv[1])
