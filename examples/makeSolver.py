import os

from robotmpcs.models.pointRobotMpcModel import PointRobotMpcModel
from robotmpcs.models.planarArmMpcModel import PlanarArmMpcModel
from robotmpcs.models.diffDriveMpcModel import DiffDriveMpcModel
from robotmpcs.models.boxerMpcModel import BoxerMpcModel


def main():
    time_horizon = 10
    dim_goal = 2
    dt = 0.01
    #mpcModel = PlanarArmMpcModel(dim_goal, 4, time_horizon)
    #mpcModel = PointRobotMpcModel(dim_goal, time_horizon)
    #mpcModel = DiffDriveMpcModel(2, time_horizon)
    mpcModel = BoxerMpcModel(time_horizon)
    mpcModel.setDt(dt)
    #mpcModel.setSlack()
    mpcModel.setObstacles(1, 2)
    mpcModel.setModel()
    mpcModel.setCodeoptions()
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpcModel.generateSolver(location=path_to_solvers)


if __name__ == "__main__":
    main()
