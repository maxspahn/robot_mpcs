from robot_mpcs.pointRobotMpcModel import PointRobotMpcModel


def main():
    mpcModel = PointRobotMpcModel(2, 2)
    mpcModel.setDt(0.5)
    mpcModel.setSlack()
    mpcModel.setObstacles(0, 2)
    mpcModel.setModel()
    solverName = "solver_n2_05_H2"
    mpcModel.setCodeoptions(solverName, debug=False)
    mpcModel.generateSolver(location="./solvers/")


if __name__ == "__main__":
    main()
