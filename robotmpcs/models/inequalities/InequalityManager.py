from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.models.inequalities.SelfCollisionAvoidanceConstraints import SelfCollisionAvoidanceConstraints
from robotmpcs.models.inequalities.JointLimitConstraints import JointLimitConstraints
from robotmpcs.models.inequalities.SpeedLimitConstraints import SpeedLimitConstraints
from robotmpcs.models.inequalities.InputLimitConstraints import InputLimitConstraints
from robotmpcs.models.inequalities.RadialConstraint import RadialConstraints
class InequalityManager(MpcBase):

    def __init__(self, ParamMap={}, inequality_list = [], **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap
        self._kwargs = kwargs
        self._inequalitiy_list = inequality_list

    def set_constraints(self):
        # update paramMap
        self.inequality_modules = [RadialConstraints(self._paramMap, **self._kwargs),
                                   SelfCollisionAvoidanceConstraints(self._paramMap, **self._kwargs),
                                   JointLimitConstraints(self._paramMap, **self._kwargs),
                                   SpeedLimitConstraints(self._paramMap, **self._kwargs),
                                   InputLimitConstraints(self._paramMap, **self._kwargs)]




    def eval_inequalities(self, z, p):
        all_ineqs = []
        for module in self.inequality_modules:
            all_ineqs += module.eval_constraint(z,p)
        if self._ns > 0:
            s = z[self._nx]
            for ineq in all_ineqs:
                ineq += s
        return all_ineqs



