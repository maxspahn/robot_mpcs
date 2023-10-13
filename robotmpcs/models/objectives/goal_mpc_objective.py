import casadi as ca
import numpy as np
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.models.utils.utils import diagSX

class GoalMpcObjective(MpcBase):

    def __init__(self, _paramMap, **kwargs):
        super().__init__(**kwargs)
        self._paramMap = _paramMap




    def eval_objectiveCommon(self, z, p):
        variables = self.extractVariables(z)
        q = variables[0]
        vel = self.get_velocity(z)
        w = p[self._paramMap["w"]]
        wvel = p[self._paramMap["wvel"]]
        g = p[self._paramMap["g"]]
        W = diagSX(w, self._m)
        Wvel = diagSX(wvel, self._nx-self._n)
        fk_ee = self._fk.fk(
            q,
            self._robot_config.root_link,
            self._robot_config.end_link,
            positionOnly=True
        )
        Jvel = ca.dot(vel, ca.mtimes(Wvel, vel))
        err = fk_ee - g
        Jx = ca.dot(err, ca.mtimes(W, err))
        Jobst = 0
        Js = 0
        obstDistances = 1/ca.vcat(self.eval_obstacleDistances(z, p))
        wobst = ca.SX(np.ones(obstDistances.shape[0]) * p[self._paramMap['wobst']])
        Wobst = diagSX(wobst, obstDistances.shape[0])
        Jobst += ca.dot(obstDistances, ca.mtimes(Wobst, obstDistances))
        if self._ns > 0:
            s = z[self._nx]
            ws = p[self._paramMap["ws"]]
            Js += ws * s ** 2
        return Jx, Jvel, Js, Jobst

    def eval_objectiveN(self, z, p):
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        return Jx + Jvel + Js + Jobst

    def eval_objective(self, z, p):
        wu = p[self._paramMap["wu"]]
        Wu = diagSX(wu, self._nu)
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        _, _, qddot, *_ = self.extractVariables(z)
        Ju = ca.dot(qddot, ca.mtimes(Wu, qddot))
        return Jx + Jvel + Js + Jobst + Ju