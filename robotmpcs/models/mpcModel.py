import os
import casadi as ca
from dataclasses import dataclass
from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import numpy as np
import sys
sys.path.append("../")
sys.path.append("")
from examples.helpers import load_forces_path
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
load_forces_path()
import forcespro
import yaml
from shutil import move, rmtree
from glob import glob
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.models.objectives.goal_mpc_objective import GoalMpcObjective









class MpcModel(MpcBase):
    def __init__(self, initParamMap=True, **kwargs):

        super().__init__(**kwargs)
        self._kwargs = kwargs

        if initParamMap:
            self._limits = {
                "x": {"low": np.ones(self._nx) * -100, "high": np.ones(self._nx) * 100},
                "u": {"low": np.ones(self._nu) * -100, "high": np.ones(self._nu) * 100},
                "s": {"low": np.zeros(1), "high": np.ones(1) * np.inf},
            }
            self.initParamMap()

    def initParamMap(self):
        self._paramMap = {}
        self._npar = 0
        self.addEntry2ParamMap("wu", self._nu)
        self.addEntry2ParamMap("wvel", self._nx-self._n)
        self.addEntry2ParamMap("w", self._m)
        if self._config.slack:
            self._ns = 1
            self.addEntry2ParamMap("ws", 1)
        self.addEntry2ParamMap("g", self._m)
        self.addEntry2ParamMap("r_body", 1)
        self.addEntry2ParamMap("lower_limits", self._n)
        self.addEntry2ParamMap("upper_limits", self._n)
        self.addEntry2ParamMap("lower_limits_vel", 2)
        self.addEntry2ParamMap("upper_limits_vel", 2)
        #self.addEntry2ParamMap("lower_limits_u", 2)
        #self.addEntry2ParamMap("upper_limits_u", 2)
        self.setObstacles()

    def addEntry2ParamMap(self, name, n_par):
        self._paramMap[name] = list(range(self._npar, self._npar + n_par))
        self._npar += n_par

    def setSelfCollisionAvoidance(self, pairs):
        self._pairs = pairs

    def setObstacles(self):
        self.addEntry2ParamMap("obst", 4 * self._config.number_obstacles)
        self.addEntry2ParamMap('wobst', 1)







    def eval_inequalities(self, z, p):
        all_ineqs = self.eval_obstacleDistances(z, p) + self.eval_jointLimits(z, p) + self.eval_selfCollision(z, p) + self.eval_speedLimits(z,p) #+ self.eval_InputLimits(z,p)
        if self._ns > 0:
            s = z[self._nx]
            for ineq in all_ineqs:
                ineq  += s
        return all_ineqs

    def eval_obstacleDistances(self, z, p):
        ineqs = []
        q, *_ = self.extractVariables(z)
        if self._ns > 0:
            s = z[self._nx]
        else:
            s = 0.0
        if "obst" in self._paramMap.keys():
            obsts = p[self._paramMap["obst"]]
            r_body = p[self._paramMap["r_body"]]
            for j, collision_link in enumerate(self._robot_config.collision_links):
                fk = self._fk.fk(
                    q,
                    self._robot_config.root_link,
                    collision_link,
                    positionOnly=True
                )[0:self._m]
                for i in range(self._config.number_obstacles):
                    obst = obsts[i * (self._m_obst + 1) : (i + 1) * (self._m_obst + 1)]
                    x = obst[0 : self._m_obst]
                    r = obst[self._m_obst]
                    dist = ca.norm_2(fk - x)
                    ineqs.append(dist - r - r_body)
        return ineqs

    def eval_selfCollision(self, z, p):
        q, *_ = self.extractVariables(z)
        r_body = p[self._paramMap["r_body"]]
        ineqs = []
        for pair in self._robot_config.selfCollision['pairs']:
            fk1 = self._fk.fk(q, self._robot_config.root_link, pair[0], positionOnly=True)[0: self._m]
            fk2 = self._fk.fk(q, self._robot_config.root_link, pair[1], positionOnly=True)[0: self._m]
            dist = ca.norm_2(fk1 - fk2)
            ineqs.append(dist - (2 * r_body))
        return ineqs

    def eval_speedLimits(self, z, p):
        # Parameters in state boundaries?
        q, qdot, _ = self.extractVariables(z)
        vel = qdot[-2:]
        lower_limits = p[self._paramMap["lower_limits_vel"]]
        upper_limits = p[self._paramMap["upper_limits_vel"]]
        ineqs = []
        for j in range(2):
            dist_lower = vel[j] - lower_limits[j]
            dist_upper = upper_limits[j] - vel[j]
            ineqs.append(dist_lower)
            ineqs.append(dist_upper)
        return ineqs

    def eval_jointLimits(self, z, p):
        # Parameters in state boundaries?
        q, *_ = self.extractVariables(z)
        lower_limits = p[self._paramMap["lower_limits"]]
        upper_limits = p[self._paramMap["upper_limits"]]
        ineqs = []
        for j in range(self._n):
            dist_lower = q[j] - lower_limits[j]
            dist_upper = upper_limits[j] - q[j]
            ineqs.append(dist_lower)
            ineqs.append(dist_upper)
        return ineqs

    def eval_InputLimits(self, z, p):
        # Parameters in state boundaries?
        u = z[-self._nu:]
        lower_limits = p[self._paramMap["lower_limits_u"]]
        upper_limits = p[self._paramMap["upper_limits_u"]]
        ineqs = []
        for j in range(2):
            dist_lower = u[j] - lower_limits[j]
            dist_upper = upper_limits[j] - u[j]
            ineqs.append(dist_lower)
            ineqs.append(dist_upper)
        return ineqs

    def setLimits(self, limits):
        self._limits = limits

    def continuous_dynamics(self, x, u):
        qdot = x[self._n: self._nx]
        qddot = u[-self._nu:]
        acc = ca.vertcat(qdot, qddot)
        return acc

    def setDt(self, dt):
        self._dt = dt

    def setModel(self):
        self.objective = GoalMpcObjective(self._paramMap, **self._kwargs)
        self._model = forcespro.nlp.SymbolicModel(self._N)
        self._model.continuous_dynamics = self.continuous_dynamics
        self._model.objective = self.objective.eval_objective
        self._model.objectiveN = self.objective.eval_objectiveN
        E = np.concatenate(
            [np.eye(self._nx), np.zeros((self._nx, self._nu + self._ns))], axis=1
        )
        self._model.E = E
        if self._ns > 0:
            self._model.lb = np.concatenate(
                (self._limits["x"]["low"], self._limits["s"]["low"], self._limits["u"]["low"])
            )
            self._model.ub = np.concatenate(
                (self._limits["x"]["high"], self._limits["s"]["high"], self._limits["u"]["high"])
            )
        else:
            self._model.lb = np.concatenate(
                (self._limits["x"]["low"], self._limits["u"]["low"])
            )
            self._model.ub = np.concatenate(
                (self._limits["x"]["high"], self._limits["u"]["high"])
            )
        self._model.npar = self._npar
        self._model.nvar = self._nx + self._nu + self._ns
        self._model.neq = self._nx
        number_inequalities = 0
        number_inequalities += self._config.number_obstacles * len(self._robot_config.collision_links)
        number_inequalities += len(self._robot_config.selfCollision['pairs'])
        number_inequalities += self._n * 2
        number_inequalities +=  2 *2
        # number_inequalities += 2 * 2
        self._model.nh = number_inequalities
        self._model.hu = np.ones(number_inequalities) * np.inf
        self._model.hl = np.zeros(number_inequalities)
        self._model.ineq = self.eval_inequalities
        self._model.xinitidx = range(0, self._nx)

    def setCodeoptions(self, **kwargs):
        solverName = self._modelName + "_n" + str(self._n) + "_" + str(self._dt).replace('.','') + "_H" + str(self._N)
        if not self._config.slack:
            solverName += "_noSlack"
        if solverName in kwargs:
            solverName = kwargs.get('solverName')
        self._solverName = solverName
        self._codeoptions = forcespro.CodeOptions(solverName)
        self._codeoptions.nlp.integrator.type = "ERK2"
        self._codeoptions.nlp.integrator.Ts = self._dt
        self._codeoptions.nlp.integrator.nodes = 5
        if self._config.debug:
            self._codeoptions.printlevel = 2
            self._codeoptions.optlevel = 0
        else:
            self._codeoptions.printlevel = 0
            self._codeoptions.optlevel = 3

    def generateSolver(self, location="./"):
        if self._config.debug:
            location += 'debug/'
        _ = self._model.generate_solver(self._codeoptions)
        with open(self._solverName + '/paramMap.yaml', 'w') as outfile:
            yaml.dump(self._paramMap, outfile, default_flow_style=False)
        properties = {"nx": self._nx, "nu": self._nu, "npar": self._npar, "ns": self._ns, "m": self._m}
        with open(self._solverName + '/properties.yaml', 'w') as outfile:
            yaml.dump(properties, outfile, default_flow_style=False)
        if os.path.exists(location + self._solverName) and os.path.isdir(location + self._solverName):
            rmtree(location + self._solverName)
        move(self._solverName, location + self._solverName)
        for file in glob(r'*.forces'):
            move(file, location)

