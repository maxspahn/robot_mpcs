import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
class VelLimitConstraints(MpcBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._n_ineq = 2 # todo adapt dimension

    def set_parameters(self, ParamMap,npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap("lower_limits_vel", 2)
        self.addEntry2ParamMap("upper_limits_vel", 2)
        return self._paramMap, self._npar


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