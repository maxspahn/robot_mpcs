import numpy as np
import yaml
import os
import sys
import forcespro
import re
import robotmpcs
from robotmpcs.models.mpcModel import MpcConfiguration



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
    def __init__(self, robotType, solversDir, **kwargs):
        self._config = MpcConfiguration(**kwargs)
        self._robotType = robotType
        """
        self._paramMap, self._npar, self._nx, self._nu, self._ns = getParameterMap(
            self_config.n, self.m(), self._config.number_obstacles, self.m(), self._config.slack
        )
        """
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

    def reset(self):
        print("RESETTING PLANNER")
        self._x0 = np.zeros(shape=(self._config.time_horizon, self._nx + self._nu + self._ns))
        self._xinit = np.zeros(self._nx)
        if self._config.slack:
            self._slack = 0.0
        self._x0[-1, -1] = 0.1
        self._params = np.zeros(shape=(self._npar * self._config.time_horizon), dtype=float)
        for i in range(self._config.time_horizon):
            self._params[
                [self._npar * i + val for val in self._paramMap["w"]]
            ] = self._config.weights["w"]
            self._params[
                [self._npar * i + val for val in self._paramMap["wvel"]]
            ] = self._config.weights["wvel"]
            self._params[
                [self._npar * i + val for val in self._paramMap["wu"]]
            ] = self._config.weights["wu"]
            if self._config.slack:
                self._params[
                    [self._npar * i + val for val in self._paramMap["ws"]]
                ] = self._config.weights["ws"]
            if 'wobst' in self._config.weights:
                self._params[
                    [self._npar * i + val for val in self._paramMap["wobst"]]
                ] = self._config.weights["wobst"]

    def m(self):
        return self._properties['m']

    def dynamic(self):
        if 'dynamic' in self._setup.keys():
            return self._setup['dynamic']
        else:
            return False


    def setObstacles(self, obsts, r_body):
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

    def setSelfCollisionAvoidance(self, r_body):
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

    def setGoal(self, goal):
        for i in range(self._config.time_horizon):
            for j in range(self.m()):
                if j >= len(goal.position()):
                    position = 0
                else:
                    position = goal.position()[j]
                self._params[self._npar * i + self._paramMap["g"][j]] = position
    def concretize(self):
        pass

    def shiftHorizon(self, output, ob):
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

    def setX0(self, xinit):
        for i in range(self._config.time_horizon):
            self._x0[i][0 : self._nx] = xinit

    def solve(self, ob):
        # print("Observation : " , ob[0:self._nx])
        self._xinit = ob[0 : self._nx]
        if ob.size > self._nx:
            self.updateDynamicObstacles(ob[self._nx:])
        action = np.zeros(self._nu)
        problem = {}
        problem["xinit"] = self._xinit
        self._x0[0][0 : self._nx] = self._xinit
        self.setX0(self._xinit)
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params
        # debug
        debug = False
        if debug:
            nbPar = int(len(self._params)/self._config.time_horizon)
            if self._config.slack:
                z = np.concatenate((self._xinit, np.array([self._slack])))
            else:
                z = self._xinit
            p = self._params[0:nbPar]
            #J = eval_obj(z, p)
            ineq = eval_ineq(z, p)
            #print("ineq : ", ineq)
            # __import__('pdb').set_trace()
            """
            for i in range(self._config.time_horizon):
                z = self._x0[i]
                ineq = eval_ineq(z, p)
            """
            #print("J : ", J)
            #print('z : ', z)
            #print('xinit : ', self._xinit)
        output, exitflag, info = self._solver.solve(problem)
        if exitflag < 0:
            print(exitflag)
        if self._config.time_horizon < 10:
            key1 = 'x1'
        elif self._config.time_horizon >= 10 and self._config.time_horizon < 100:
            key1 = 'x01'
        elif self._config.time_horizon >= 100:
            key1 = 'x001'
        action = output[key1][-self._nu :]
        if self._config.slack:
            self._slack = output[key1][self._nx]
            if self._slack > 1e-3:
                print("slack : ", self._slack)
        # print('action : ', action)
        # print("prediction : ", output["x02"][0:self._nx])
        self.shiftHorizon(output, ob)
        return action, info

    def concretize(self):
        self._actionCounter = self._config.interval

    def computeAction(self, *args):
        ob = np.concatenate(args)
        if self._actionCounter >= self._config.interval:
            self._action, info = self.solve(ob)
            self._actionCounter = 1
        else:
            self._actionCounter += 1
        return self._action

