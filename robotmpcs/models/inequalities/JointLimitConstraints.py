import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
class JointLimitConstraints(MpcBase):

    def __init__(self, ParamMap={}, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap

    def get_number_ineq(self):
        return self._n*2

    def eval_constraint(self, z, p):
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