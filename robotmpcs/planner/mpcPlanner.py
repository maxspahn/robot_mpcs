import numpy as np
import yaml
import os
import sys
import forcespro
import re
import robotmpcs






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
    def __init__(self, setupFile, robotType, solversDir):
        required_keys = ["type", "dof", "obst", "weights", "interval", "time_horizon", "dt"]
        self._required_keys = required_keys
        self._setupFile = setupFile
        self._robotType = robotType
        self.parseSetup()
        """
        self._paramMap, self._npar, self._n_state, self._n_control_input, self._n_slack = getParameterMap(
            self.n(), self.m(), self.nbObstacles(), self.m(), self.useSlack()
        )
        """
        dt_str = str(self.dt()).replace(".", "")
        self._solverFile = (
            solversDir
            + self._robotType
            + "_n" + str(self.dof())
            + "_"
            + dt_str
            + "_H"
            + str(self.timeHorizon())
        )
        if not self.useSlack():
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
        self._n_state = self._properties['n_state']
        self._n_control_input = self._properties['n_control_input']
        self._n_slack = self._properties['n_slack']
        self._npar = self._properties['npar']
        try:
            print("Loading solver %s" % self._solverFile)
            self._solver = forcespro.nlp.Solver.from_directory(self._solverFile)
        except Exception as e:
            print("FAILED TO LOAD SOLVER")
            raise e

    def parseSetup(self):
        with open(self._setupFile, "r") as setupStream:
            self._setup = yaml.safe_load(setupStream)
        self.checkCompleteness()

    def checkCompleteness(self):
        incomplete = False
        missingKeys = ""
        for key in self._required_keys:
            if key not in self._setup.keys():
                incomplete = True
                missingKeys += key + ", "
        if incomplete:
            raise PlannerSettingIncomplete("Missing keys: %s" % missingKeys[:-2])

    def plannerType(self):
        return self._setup['type']

    def reset(self):
        print("RESETTING PLANNER")
        self._x0 = np.zeros(shape=(self.timeHorizon(), self._n_state + self._n_control_input + self._n_slack))
        self._xinit = np.zeros(self._n_state)
        if self.useSlack():
            self._slack = 0.0
        self._x0[-1, -1] = 0.1
        self._params = np.zeros(shape=(self._npar * self.timeHorizon()), dtype=float)
        for i in range(self.timeHorizon()):
            self._params[
                [self._npar * i + val for val in self._paramMap["w"]]
            ] = self.weights()["w"]
            self._params[
                [self._npar * i + val for val in self._paramMap["wvel"]]
            ] = self.weights()["wvel"]
            self._params[
                [self._npar * i + val for val in self._paramMap["wu"]]
            ] = self.weights()["wu"]
            if self.useSlack():
                self._params[
                    [self._npar * i + val for val in self._paramMap["ws"]]
                ] = self.weights()["ws"]
            if 'wobst' in self.weights():
                self._params[
                    [self._npar * i + val for val in self._paramMap["wobst"]]
                ] = self.weights()["wobst"]

    def dimGoal(self):
        return 2

    def interval(self):
        return self._setup["interval"]

    def useSlack(self):
        if 'slack' in self._setup.keys():
            return self._setup['slack']
        else:
            return True

    def dof(self):
        return self._setup["dof"]

    def timeHorizon(self):
        return self._setup["time_horizon"]

    def dt(self):
        return self._setup["dt"]

    def dynamic(self):
        if 'dynamic' in self._setup.keys():
            return self._setup['dynamic']
        else:
            return False

    def weights(self):
        return self._setup["weights"]

    def nbObstacles(self):
        return self._setup["obst"]["nbObst"]

    def setObstacles(self, obsts, r_body):
        self._r = 0.1
        for i in range(self.timeHorizon()):
            self._params[self._npar * i + self._paramMap["r_body"][0]] = r_body
            for j in range(self.nbObstacles()):
                if j < len(obsts):
                    obst = obsts[j]
                else:
                    obst = EmptyObstacle()
                for m_i in range(obst.dim()):
                    paramsIndexObstX = self._npar * i + self._paramMap['obst'][j * (self.dimGoal() + 1) + m_i]
                    self._params[paramsIndexObstX] = obst.position()[m_i]
                paramsIndexObstR = self._npar * i + self._paramMap['obst'][j * (self.dimGoal() + 1) + self.dimGoal()]
                self._params[paramsIndexObstR] = obst.radius()

    def updateDynamicObstacles(self, obstArray):
        nbDynamicObsts = int(obstArray.size / 3 / self.dimGoal())
        for j in range(self.nbObstacles()):
            if j < nbDynamicObsts:
                obstPos = obstArray[:self.dimGoal()]
                obstVel = obstArray[self.dimGoal():2 * self.dimGoal()]
                obstAcc = obstArray[2 * self.dimGoal():3 * self.dimGoal()]
            else:
                obstPos = np.ones(self.dimGoal()) * -100
                obstVel = np.zeros(self.dimGoal())
                obstAcc = np.zeros(self.dimGoal())
            for i in range(self.timeHorizon()):
                for m_i in range(self.dimGoal()):
                    paramsIndexObstX = self._npar * i + self._paramMap['obst'][j * (self.dimGoal() + 1) + m_i]
                    predictedPosition = obstPos[m_i] + obstVel[m_i] * self.dt() * i + 0.5 * (self.dt() * i)**2 * obstAcc[m_i]
                    self._params[paramsIndexObstX] = predictedPosition
                paramsIndexObstR = self._npar * i + self._paramMap['obst'][j * (self.dimGoal() + 1) + self.dimGoal()]
                self._params[paramsIndexObstR] = self._r

    def setSelfCollisionAvoidance(self, r_body):
        for i in range(self.timeHorizon()):
            self._params[self._npar * i + self._paramMap["r_body"][0]] = r_body

    def setJointLimits(self, limits):
        for i in range(self.timeHorizon()):
            for j in range(self.dof()):
                self._params[
                    self._npar * i + self._paramMap["lower_limits"][j]
                ] = limits[0][j]
                self._params[
                    self._npar * i + self._paramMap["upper_limits"][j]
                ] = limits[1][j]

    def setGoal(self, goal):
        for i in range(self.timeHorizon()):
            for j in range(self.dimGoal()):
                self._params[self._npar * i + self._paramMap["g"][j]] = goal.position()[j]

    def concretize(self):
        pass

    def shiftHorizon(self, output, ob):
        for key in output.keys():
            if self.timeHorizon() < 10:
                stage = int(key[-1:])
            elif self.timeHorizon() >= 10 and self.timeHorizon() < 100:
                stage = int(key[-2:])
            elif self.timeHorizon() >= 100:
                stage = int(key[-3:])
            if stage == 1:
                continue
            self._x0[stage - 2, 0 : len(output[key])] = output[key]

    def setX0(self, xinit):
        for i in range(self.timeHorizon()):
            self._x0[i][0 : self._n_state] = xinit

    def solve(self, ob):
        # print("Observation : " , ob[0:self._n_state])
        self._xinit = ob[0 : self._n_state]
        if ob.size > self._n_state:
            self.updateDynamicObstacles(ob[self._n_state:])
        action = np.zeros(self._n_control_input)
        problem = {}
        problem["xinit"] = self._xinit
        self._x0[0][0 : self._n_state] = self._xinit
        self.setX0(self._xinit)
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params
        # debug
        debug = False
        if debug:
            nbPar = int(len(self._params) / self.timeHorizon())
            if self.useSlack():
                z = np.concatenate((self._xinit, np.array([self._slack])))
            else:
                z = self._xinit
            p = self._params[0:nbPar]
            #J = eval_obj(z, p)
            ineq = eval_ineq(z, p)
            #print("ineq : ", ineq)
            # __import__('pdb').set_trace()
            """
            for i in range(self.H()):
                z = self._x0[i]
                ineq = eval_ineq(z, p)
            """
            #print("J : ", J)
            #print('z : ', z)
            #print('xinit : ', self._xinit)
        output, exitflag, info = self._solver.solve(problem)
        if exitflag < 0:
            print(exitflag)
        if self.timeHorizon() < 10:
            key1 = 'x1'
        elif self.timeHorizon() >= 10 and self.timeHorizon() < 100:
            key1 = 'x01'
        elif self.timeHorizon() >= 100:
            key1 = 'x001'
        action = output[key1][-self._n_control_input:]
        if self.useSlack():
            self._slack = output[key1][self._n_state]
            if self._slack > 1e-3:
                print("slack : ", self._slack)
        # print('action : ', action)
        # print("prediction : ", output["x02"][0:self._n_state])
        self.shiftHorizon(output, ob)
        return action, info

    def concretize(self):
        self._actionCounter = self.interval()

    def computeAction(self, *args):
        ob = np.concatenate(args)
        if self._actionCounter >= self.interval():
            self._action, info = self.solve(ob)
            self._actionCounter = 1
        else:
            self._actionCounter += 1
        return self._action

