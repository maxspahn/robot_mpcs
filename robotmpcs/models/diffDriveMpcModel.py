from robotmpcs.models.mpcModel import MpcModel, diagSX
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
        self._fk = FkCreator('groundRobot', self._n).fk()
        self.initParamMap()
        self._modelName = "diffDrive"

    def extractVariables(self, z):
        q = z[0: self._n]
        qdot = z[self._n:2 * self._n]
        vel_rel = z[2*self._n: 2*self._n + self._nu]
        qddot = z[-self._nu:]
        return q, qdot, qddot, vel_rel

    def eval_objectiveCommon(self, z, p):
        q, qdot, _, vel = self.extractVariables(z)
        w = p[self._paramMap["w"]]
        wvel = p[self._paramMap["wvel"]]
        g = p[self._paramMap["g"]]
        W = diagSX(w, self._m)
        Wvel = diagSX(wvel, self._nu)
        fk_ee = self._fk.fk(q, self._n, positionOnly=True)[0:self._m]
        Jvel = ca.dot(vel, ca.mtimes(Wvel, vel))
        err = fk_ee - g
        Jx = ca.dot(err, ca.mtimes(W, err))
        Jobst = 0
        Js = 0
        if self._obstaclesInCosts:
            obstDistances = 1/ca.vcat(self.eval_obstacleDistances(z, p) )
            wobst = ca.SX(np.ones(obstDistances.shape[0]) * p[self._paramMap['wobst']])
            Wobst = diagSX(wobst, obstDistances.shape[0])
            Jobst += ca.dot(obstDistances, ca.mtimes(Wobst, obstDistances))
        if self._ns > 0:
            s = z[self._nx]
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

