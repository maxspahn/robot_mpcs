import os

from robotmpcs.pointRobotMpcModel import PointRobotMpcModel
from robotmpcs.planarArmMpcModel import PlanarArmMpcModel


def main():
    N = 2
    n = 5
    m = 2
    dt = 0.5
    mpcModel = PlanarArmMpcModel(m, N, n)
    #mpcModel = PointRobotMpcModel(m, N)
    mpcModel.setDt(dt)
    mpcModel.setSlack()
    mpcModel.setObstacles(0, 2)
    mpcModel.setModel()
    solverName = "solver_n" + str(n) + "_" + str(dt).replace('.','') + "_H" + str(N)
    mpcModel.setCodeoptions(solverName, debug=False)
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpcModel.generateSolver(location=path_to_solvers)


if __name__ == "__main__":
    main()
