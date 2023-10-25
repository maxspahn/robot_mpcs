from typing import Tuple
import numpy as np
import yaml
import os
import forcespro
from robotmpcs.models.mpcBase import MpcConfiguration



class SolverDoesNotExistError(Exception):
    def __init__(self, solverName):
        super().__init__()
        self._solverName = solverName

    def __str__(self):
        return f"Solver with name {self._solverName} does not exist."

class EmptyObstacle():
    def position(self):
        return [-100, -100, -100]

    def radius(self):
        return -100

    def dim(self):
        return 3

class PlannerSettingIncomplete(Exception):
    pass

class MPCPlanner(object):
    def __init__(self, robotType, solversDir, mpc_model=None, debug=False, **kwargs):
        self._config = MpcConfiguration(**kwargs)
        self._robotType = robotType
        self._initial_step = True

        """
        self._paramMap, self._npar, self._nx, self._nu, self._ns = getParameterMap(
            self_config.n, self.m(), self._config.number_obstacles, self.m(), self._config.slack
        )
        """
        self._debug = debug
        dt_str = str(self._config.time_step).replace(".", "")
        self._solverFile = (
            solversDir
            + self._robotType
            + "_n" + str(self._config.n)
            + "_"
            + dt_str
            + "_H"
            + str(self._config.time_horizon)
        )
        if not self._config.slack:
            self._solverFile += "_noSlack"
        if not os.path.isdir(self._solverFile):
            raise(SolverDoesNotExistError(self._solverFile))
        with open(self._solverFile + "/paramMap.yaml", "r") as stream:
            try:
                self._paramMap = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(self._solverFile + "/properties.yaml", "r") as stream:
            try:
                self._properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self._nx = self._properties['nx']
        self._nu = self._properties['nu']
        self._ns = self._properties['ns']
        self._npar = self._properties['npar']
        try:
            print("Loading solver %s" % self._solverFile)
            self._solver = forcespro.nlp.Solver.from_directory(self._solverFile)
        except Exception as e:
            print("FAILED TO LOAD SOLVER")
            raise e

        if self._debug:
            self._mpc_model = mpc_model



    def reset(self):
        print("RESETTING PLANNER")
        self._x0 = np.zeros(shape=(self._config.time_horizon, self._nx + self._nu + self._ns))
        self._xinit = np.zeros(self._nx)
        self._initial_step = True
        if self._config.slack:
            self._slack = 0.0
        #self._x0[-1, -1] = 0.1 # todo why?
        self._params = np.zeros(shape=(self._npar * self._config.time_horizon), dtype=float)
        for i in range(self._config.time_horizon):
            self._params[
                [self._npar * i + val for val in self._paramMap["wgoal"]]
            ] = self._config.weights["w"]
            # for j, val in enumerate(self._paramMap["wvel"]):
            #     self._params[[self._npar * i + val]] = self._config.weights["wvel"][j]
            self._params[
                [self._npar * i + val for val in self._paramMap["wu"]]
            ] = self._config.weights["wu"]
            if self._config.slack:
                self._params[
                    [self._npar * i + val for val in self._paramMap["ws"]]
                ] = self._config.weights["ws"]
            # if 'wobst' in self._config.weights:
            #     self._params[
            #         [self._npar * i + val for val in self._paramMap["wobst"]]
            #     ] = self._config.weights["wobst"]

    def m(self):
        return self._properties['m']

    def dynamic(self):
        if 'dynamic' in self._setup.keys():
            return self._setup['dynamic']
        else:
            return False


    def setRadialConstraints(self, obsts, r_body): #todo: need to be updated
        self._r = 0.1
        for i in range(self._config.time_horizon):
            self._params[self._npar * i + self._paramMap["r_body"][0]] = r_body
            for j in range(self._config.number_obstacles):
                if j < len(obsts):
                    obst = obsts[j]
                else:
                    obst = EmptyObstacle()
                for m_i in range(obst.dimension()):
                    paramsIndexObstX = self._npar * i + self._paramMap['obst'][j * (self.m() + 1) + m_i]
                    self._params[paramsIndexObstX] = obst.position()[m_i]
                paramsIndexObstR = self._npar * i + self._paramMap['obst'][j * (self.m() + 1) + self.m()]
                self._params[paramsIndexObstR] = obst.radius()

    def setLinearConstraints(self, lin_constr, r_body):
        for j in range(self._config.time_horizon):
            self._params[self._npar * j + self._paramMap["r_body"][0]] = r_body
            for i in range(self._config.number_obstacles):
                for m in range(4):
                    idx = self._npar * j + self._paramMap["lin_constrs_" + str(i)][m]
                    self._params[idx] = lin_constr[j][i][m]


    def updateDynamicObstacles(self, obstArray):
        nbDynamicObsts = int(obstArray.size / 3 / self.m())
        for j in range(self._config.number_obstacles):
            if j < nbDynamicObsts:
                obstPos = obstArray[:self.m()]
                obstVel = obstArray[self.m():2*self.m()]
                obstAcc = obstArray[2*self.m():3*self.m()]
            else:
                obstPos = np.ones(self.m()) * -100
                obstVel = np.zeros(self.m())
                obstAcc = np.zeros(self.m())
            for i in range(self._config.time_horizon):
                for m_i in range(self.m()):
                    paramsIndexObstX = self._npar * i + self._paramMap['obst'][j * (self.m() + 1) + m_i]
                    predictedPosition = obstPos[m_i] + obstVel[m_i] * self.dt() * i + 0.5 * (self.dt() * i)**2 * obstAcc[m_i]
                    self._params[paramsIndexObstX] = predictedPosition
                paramsIndexObstR = self._npar * i + self._paramMap['obst'][j * (self.m() + 1) + self.m()]
                self._params[paramsIndexObstR] = self._r

    def setSelfCollisionAvoidanceConstraints(self, r_body):
        for i in range(self._config.time_horizon):
            self._params[self._npar * i + self._paramMap["r_body"][0]] = r_body

    def setJointLimits(self, limits):
        for i in range(self._config.time_horizon):
            for j in range(self._config.n):
                self._params[
                    self._npar * i + self._paramMap["lower_limits"][j]
                ] = limits[0][j]
                self._params[
                    self._npar * i + self._paramMap["upper_limits"][j]
                ] = limits[1][j]

    def setVelLimits(self, limits_vel):
        for i in range(self._config.time_horizon):
            for j in range(2): # todo make dependent on vel dim
                self._params[
                    self._npar * i + self._paramMap["lower_limits_vel"][j]
                ] = limits_vel[0][j]
                self._params[
                    self._npar * i + self._paramMap["upper_limits_vel"][j]
                ] = limits_vel[1][j]

    def setInputLimits(self, limits_u):
        for i in range(self._config.time_horizon):
            for j in range(self._nu):
                self._params[
                    self._npar * i + self._paramMap["lower_limits_u"][j]
                ] = limits_u[0][j]
                self._params[
                    self._npar * i + self._paramMap["upper_limits_u"][j]
                ] = limits_u[1][j]

    def setGoalReaching(self, goal):
        for i in range(self._config.time_horizon):
            for j in range(self.m()):
                if j >= len(goal.primary_goal().position()):
                    position = 0
                else:
                    position = goal.primary_goal().position()[j]
                self._params[self._npar * i + self._paramMap["goal"][j]] = position

    def setConstraintAvoidance(self):
        for i in range(self._config.time_horizon):
            self._params[
                [self._npar * i + val for val in self._paramMap["wconstr"]]
            ] = self._config.weights["wconstr"]

    def concretize(self):
        pass

    def shiftHorizon(self, output):
        for key in output.keys():
            if self._config.time_horizon < 10:
                stage = int(key[-1:])
            elif self._config.time_horizon >= 10 and self._config.time_horizon < 100:
                stage = int(key[-2:])
            elif self._config.time_horizon >= 100:
                stage = int(key[-3:])
            if stage == 1:
                continue
            self._x0[stage - 2, 0 : len(output[key])] = output[key]
            self._x0[-1:, 0 : len(output[key])] = output[key]

    def setX0(self, initialize_type="current_state", initial_step= True):
        if initialize_type == "current_state" or initialize_type == "previous_plan" and initial_step:
            for i in range(self._config.time_horizon):
                self._x0[i][0 : self._nx] = self._xinit
                self._initial_step = False
        elif initialize_type == "previous_plan":
            self.shiftHorizon(self.output)
        else:
            np.zeros(shape=(self._config.time_horizon, self._nx + self._nu + self._ns))



    def solve(self, ob):
        # print("Observation : " , ob[0:self._nx])
        self._xinit = ob[0 : self._nx]
        if ob.size > self._nx:
            self.updateDynamicObstacles(ob[self._nx:])
        action = np.zeros(self._nu)
        problem = {}
        problem["xinit"] = self._xinit
        self.setX0(initialize_type=self._config.initialization, initial_step=self._initial_step)
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params
        # debug

        if self._debug:
            print('debugging')
            z = problem["xinit"]
            p = problem["all_parameters"]
            j = 1
            ineq = self._mpc_model._model.ineq[j](z, p)
            print(self._config.constraints)
            print("Inequalities: {}".format(ineq))

        self.output, exitflag, info = self._solver.solve(problem)
        if exitflag < 0:
            print(exitflag)
        if self._config.time_horizon < 10:
            key0 = 'x1'
            key1 = 'x2'
        elif self._config.time_horizon >= 10 and self._config.time_horizon < 100:
            key0 = 'x01'
            key1 = 'x02'
        elif self._config.time_horizon >= 100:
            key0 = 'x001'
            key1 = 'x002'
        # If in velocity mode, the action should be velocities instead of accelerations
        if self._config.control_mode == "vel":
            action = self.output[key1][-self._nu-self._nu: -self._nu]
        elif self._config.control_mode == "acc":
            action = self.output[key0][-self._nu:]
        else:
            print("No valid control mode specified!")
            action = np.zeros((self._nu))
        if self._config.slack:
            self._slack = self.output[key0][self._nx]
            if self._slack > 1e-3:
                print("slack : ", self._slack)


        return action, self.output, info, exitflag

    def concretize(self):
        self._actionCounter = self._config.interval

    def computeAction(self, *args) -> Tuple[np.ndarray, dict, int]:
        ob = np.concatenate(args[:3])

        if self._actionCounter >= self._config.interval:
            self._action, output, info, exitflag = self.solve(ob)
            self._actionCounter = 1
        else:
            self._actionCounter += 1
        return self._action, output, exitflag

