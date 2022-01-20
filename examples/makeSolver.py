import os

from robotmpcs.pointRobotMpcModel import PointRobotMpcModel
from robotmpcs.planarArmMpcModel import PlanarArmMpcModel
from robotmpcs.diffDriveMpcModel import DiffDriveMpcModel


def main():
    N = 10
    m = 2
    dt = 0.01
    #mpcModel = PlanarArmMpcModel(m, N, 4)
    mpcModel = PointRobotMpcModel(m, N)
    #mpcModel = DiffDriveMpcModel(2, N)
    mpcModel.setDt(dt)
    #mpcModel.setSlack()
    mpcModel.setObstacles(1, 2)
    mpcModel.setModel()
    mpcModel.setCodeoptions()
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpcModel.generateSolver(location=path_to_solvers)


if __name__ == "__main__":
    main()
