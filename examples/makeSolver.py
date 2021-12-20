from robot_mpcs.pointRobotMpcModel import PointRobotMpcModel


def main():
    mpcModel = PointRobotMpcModel(2, 10)
    mpcModel.setDt(0.5)
    mpcModel.setSlack()
    mpcModel.setObstacles(5, 2)
    mpcModel.setModel()
    solverName = "solver_n2_05_H10_noSlack"
    mpcModel.setCodeoptions(solverName, debug=True)
    mpcModel.generateSolver(location="./solvers/")


if __name__ == "__main__":
    main()
