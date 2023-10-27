import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.utils.utils import diagSX

class GoalReaching(MpcBase):

    def __init__(self,ineq_modules, **kwargs):
        super().__init__(**kwargs)

    def set_parameters(self, ParamMap, npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap("goal", self._m)
        self.addEntry2ParamMap("wgoal", self._m)
        return self._paramMap, self._npar


    def eval_objective(self, z, p):
        variables = self.extractVariables(z)
        q = variables[0]
        pos_ee = self._fk.fk(
            q,
            self._robot_config.root_link,
            self._robot_config.end_link,
            positionOnly=True
        )
        goal = p[self._paramMap["goal"]]
        w = p[self._paramMap["wgoal"]]
        W = diagSX(w, self._m)
        err = pos_ee - goal
        Jgoal = ca.dot(err, ca.mtimes(W, err))

        return Jgoal