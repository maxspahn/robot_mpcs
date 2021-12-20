from robot_mpcs.pointRobotMpcModel import PointRobotMpcModel


def main():
    mpcModel = PointRobotMpcModel(2, 10)
    mpcModel.setDt(0.01)
    mpcModel.setObstacles(5, 2)
    mpcModel.setModel()
    solverName = "pointMass_dt001_N10_ns0"
    mpcModel.setCodeoptions(solverName, debug=True)
    mpcModel.generateSolver()


if __name__ == "__main__":
    main()
