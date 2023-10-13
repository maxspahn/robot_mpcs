from robotmpcs.models.mpcBase import MpcBase
class InputLimitConstraints(MpcBase):

    def __init__(self, ParamMap={}, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap


    def eval_constraint(self, z, p):
        # Parameters in state boundaries?
        u = z[-self._nu:]
        lower_limits = p[self._paramMap["lower_limits_u"]]
        upper_limits = p[self._paramMap["upper_limits_u"]]
        ineqs = []
        for j in range(2):
            dist_lower = u[j] - lower_limits[j]
            dist_upper = upper_limits[j] - u[j]
            ineqs.append(dist_lower)
            ineqs.append(dist_upper)
        return ineqs
