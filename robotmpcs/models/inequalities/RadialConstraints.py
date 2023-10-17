import casadi as ca
import sys
from robotmpcs.models.mpcBase import MpcBase
class RadialConstraints(MpcBase):

    def __init__(self, ParamMap={}, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap

    def get_number_ineq(self):
        return self._config.number_obstacles * len(self._robot_config.collision_links)


    def eval_constraint(self, z, p):
        ineqs = self.eval_obstacleDistances(z,p)

        return ineqs

