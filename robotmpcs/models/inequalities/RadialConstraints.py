import casadi as ca
import sys
from robotmpcs.models.mpcBase import MpcBase
class RadialConstraints(MpcBase):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self._n_ineq = self._config.number_obstacles * len(self._robot_config.collision_links)

    def set_parameters(self, ParamMap, npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap("r_body", 1)
        self.addEntry2ParamMap("obst", 4 * self._config.number_obstacles)

        return self._paramMap, self._npar


    def eval_constraint(self, z, p):
        ineqs = self.eval_obstacleDistances(z,p,j)
        return ineqs

