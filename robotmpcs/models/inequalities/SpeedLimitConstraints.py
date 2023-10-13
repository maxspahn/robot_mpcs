import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
class SpeedLimitConstraints(MpcBase):

    def __init__(self, ParamMap={}, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap


    def eval_constraint(self, z, p):
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