import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
class SelfCollisionAvoidanceConstraints(MpcBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._n_ineq = len(self._robot_config.selfCollision['pairs'])

    def set_parameters(self, ParamMap, npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap("r_body", 1)

        return self._paramMap, self._npar

    def eval_constraint(self, z, p):
        q, *_ = self.extractVariables(z)
        r_body = p[self._paramMap["r_body"]]
        ineqs = []
        for pair in self._robot_config.selfCollision['pairs']:
            fk1 = self._fk.fk(q, self._robot_config.root_link, pair[0], positionOnly=True)[0: self._m]
            fk2 = self._fk.fk(q, self._robot_config.root_link, pair[1], positionOnly=True)[0: self._m]
            dist = ca.norm_2(fk1 - fk2)
            ineqs.append(dist - (2 * r_body))
        return ineqs
