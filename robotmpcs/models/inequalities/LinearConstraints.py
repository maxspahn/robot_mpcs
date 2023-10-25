from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.utils.utils import point_to_plane
class LinearConstraints(MpcBase):
    '''
    takes plane defined as ax + by + cz + d = 0 to set linear constraints
    '''

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self._n_ineq = self._config.number_obstacles * len(self._robot_config.collision_links)

    def set_parameters(self, ParamMap, npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap("r_body", 1)
        for j in range(self._N):
            for i in range(self._config.number_obstacles):
                    self.addEntry2ParamMap("lin_constrs_" + str(i), 4)

        return self._paramMap, self._npar


    def eval_constraint(self, z, p):
        ineqs = []
        q, *_ = self.extractVariables(z)
        r_body = p[self._paramMap["r_body"]]
        for l, collision_link in enumerate(self._robot_config.collision_links):
            pos = self._fk.fk(
                q,
                self._robot_config.root_link,
                collision_link,
                positionOnly=True
            )[0:self._m]
            for i in range(self._config.number_obstacles):
                    lin_constrs = p[self._paramMap["lin_constrs_" + str(i)]]
                    dist = point_to_plane(point=pos, plane=lin_constrs)
                    ineqs.append(dist - r_body)
        return ineqs
