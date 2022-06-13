from robotmpcs.models.mpcModel import MpcModel, diagSX
from forwardkinematics.fksCommon.fk_creator import FkCreator
import casadi as ca
import numpy as np


class DiffDriveMpcModel(MpcModel):

    def __init__(self, dim_goal, time_horizon):
        dof = 3
        super().__init__(dim_goal, dof, time_horizon, initParamMap=False)
        self._n_state = 8  # [x, y, theta, vel_rel0, vel_rel1, xdot, ydot, thetadot]
        self._n_control_input = 2  # [a_l, a_r]
        self._limits = {
            "x": {"low": np.ones(self._n_state) * -100, "high": np.ones(self._n_state) * 100},
            "u": {"low": np.ones(self._n_control_input) * -100, "high": np.ones(self._n_control_input) * 100},
            "s": {"low": np.zeros(1), "high": np.ones(1) * np.inf},
        }
        self._fk = FkCreator('groundRobot', self._n_dof).fk()
        self.initParamMap()
        self._modelName = "diffDrive"

    def extractVariables(self, z):
        q = z[0: self._n_dof]
        qdot = z[self._n_dof:2 * self._n_dof]
        vel_rel = z[2 * self._n_dof: 2 * self._n_dof + self._n_control_input]
        qddot = z[-self._n_control_input:]
        return q, qdot, qddot, vel_rel

    def eval_objectiveCommon(self, z, p):
        q, qdot, _, vel = self.extractVariables(z)
        w = p[self._paramMap["w"]]
        wvel = p[self._paramMap["wvel"]]
        g = p[self._paramMap["g"]]
        W = diagSX(w, self._dim_goal)
        Wvel = diagSX(wvel, self._n_control_input)
        fk_ee = self._fk.fk(q, self._n_dof, positionOnly=True)[0:self._dim_goal]
        Jvel = ca.dot(vel, ca.mtimes(Wvel, vel))
        err = fk_ee - g
        Jx = ca.dot(err, ca.mtimes(W, err))
        Jobst = 0
        Js = 0
        if self._obstaclesInCosts:
            obstDistances = 1 / ca.vcat(self.eval_obstacleDistances(z, p))
            wobst = ca.SX(np.ones(obstDistances.shape[0]) * p[self._paramMap['wobst']])
            Wobst = diagSX(wobst, obstDistances.shape[0])
            Jobst += ca.dot(obstDistances, ca.mtimes(Wobst, obstDistances))
        if self._n_slack > 0:
            s = z[self._n_state]
            ws = p[self._paramMap["ws"]]
            Js += ws * s ** 2
        return Jx, Jvel, Js, Jobst

    def computeXdot(self, x, vel):
        assert x.size() == (3, 1)
        assert vel.size() == (2, 1)
        xdot = ca.vertcat(
            ca.cos(x[2]) * vel[0],
            ca.sin(x[2]) * vel[0],
            vel[1],
        )
        return xdot

    def continuous_dynamics(self, x, u):
        # state = [x, y, theta, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        vel = x[6:8]
        xdot = self.computeXdot(x_pos, vel)
        veldot = u
        xddot = ca.SX(np.zeros(3))
        state_dot = ca.vertcat(xdot, xddot, veldot)
        return state_dot
