import os

from robotmpcs.pointRobotMpcModel import PointRobotMpcModel
from robotmpcs.planarArmMpcModel import PlanarArmMpcModel
from robotmpcs.diffDriveMpcModel import DiffDriveMpcModel


def main():
    N = 2
    m = 2
    dt = 0.5
    #mpcModel = PlanarArmMpcModel(m, N, n)
    #mpcModel = PointRobotMpcModel(m, N)
    mpcModel = DiffDriveMpcModel(2, N)
    mpcModel.setDt(dt)
    mpcModel.setSlack()
    mpcModel.setObstacles(0, 2)
    mpcModel.setModel()
    solverName = "diffDrive_solver_n" + str(mpcModel._n) + "_" + str(dt).replace('.','') + "_H" + str(N)
    mpcModel.setCodeoptions(solverName, debug=False)
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpcModel.generateSolver(location=path_to_solvers)


if __name__ == "__main__":
    main()
