import os
import casadi as ca
from dataclasses import dataclass
from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import numpy as np
import forcespro
import yaml
from shutil import move, rmtree
from glob import glob
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.models.objectives.goal_mpc_objective import GoalMpcObjective
from robotmpcs.models.inequalities.InequalityManager import InequalityManager
from robotmpcs.models.objectives.ObjectiveManager import ObjectiveManager

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
        self._inequality_manager = InequalityManager(self._paramMap, self._npar, **kwargs)
        self._paramMap, self._npar = self._inequality_manager.set_constraints()
        self.number_inequalities = 0
        for ineq_module in self._inequality_manager.inequality_modules:
            self.number_inequalities += ineq_module._n_ineq

        self._objective_manager = ObjectiveManager(self._paramMap, self._npar, self._inequality_manager.inequality_modules, **kwargs)
        self._paramMap, self._npar = self._objective_manager.set_objectives()



    def initParamMap(self):
        self._paramMap = {}
        self._npar = 0




    def setSelfCollisionAvoidance(self, pairs):
        self._pairs = pairs


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
        self._model = forcespro.nlp.SymbolicModel(self._N)
        self._model.continuous_dynamics = self.continuous_dynamics

        self._model.ineq = self._inequality_manager.eval_inequalities

        self._model.nh = self.number_inequalities
        self._model.hu = np.ones(self.number_inequalities) * np.inf
        self._model.hl = np.zeros(self.number_inequalities)


        self._model.objective = self._objective_manager.eval_objectives
        self._model.objectiveN = self._objective_manager.eval_objectiveN
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
        _ = self._model.generate_solver(self._codeoptions)
        if self._debug:
            location += 'debug/'
        with open(self._solverName + '/paramMap.yaml', 'w') as outfile:
            yaml.dump(self._paramMap, outfile, default_flow_style=False)
        properties = {"nx": self._nx, "nu": self._nu, "npar": self._npar, "ns": self._ns, "m": self._m, "constraints": self._inequality_manager.inequality_modules_strs}
        with open(self._solverName + '/properties.yaml', 'w') as outfile:
            yaml.dump(properties, outfile, default_flow_style=False)
        if os.path.exists(location + self._solverName) and os.path.isdir(location + self._solverName):
            rmtree(location + self._solverName)
        move(self._solverName, location + self._solverName)
        for file in glob(r'*.forces'):
            move(file, location)

