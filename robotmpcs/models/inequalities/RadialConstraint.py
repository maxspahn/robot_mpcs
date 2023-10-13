import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
class RadialConstraints(MpcBase):

    def __init__(self, ParamMap={}, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap

    def eval_constraint(self, z, p):
        ineqs = self.eval_obstacleDistances(z,p)

        return ineqs