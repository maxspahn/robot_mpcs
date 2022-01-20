from robotmpcs.mpcModel import MpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator
import casadi as ca
import numpy as np


class DiffDriveMpcModel(MpcModel):

    def __init__(self, m, N):
        n = 3
        super().__init__(m, n, N, initParamMap=False)
        self._nx = 8 # [x, y, theta, vel_rel0, vel_rel1, xdot, ydot, thetadot]
        self._nu = 2 # [a_l, a_r]
        self._limits = {
            "x": {"low": np.ones(self._nx) * -100, "high": np.ones(self._nx) * 100},
            "u": {"low": np.ones(self._nu) * -100, "high": np.ones(self._nu) * 100},
            "s": {"low": np.zeros(1), "high": np.ones(1) * np.inf},
        }
        self._fk = FkCreator('groundRobot', n).fk()
        self.initParamMap()
        self._modelName = "diffDrive"

    def extractVariables(self, z):
        q = z[0: self._n]
        qdot = z[self._n + self._nu: 2 * self._n + self._nu]
        vel_rel = z[self._n: self._n + self._nu]
        qddot = z[-self._nu:]
        return q, qdot, qddot, vel_rel

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
        vel = x[3:5]
        xdot = self.computeXdot(x_pos, vel)
        veldot = u
        xddot = ca.SX(np.zeros(3))
        state_dot = ca.vertcat(xdot, veldot, xddot)
        return state_dot

