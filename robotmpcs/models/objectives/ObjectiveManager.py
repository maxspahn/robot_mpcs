import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.utils.utils import diagSX

class ObjectiveManager(MpcBase):
    def __init__(self, ParamMap={}, npar=0, ineq_modules=[], **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap
        self._npar = npar
        self._kwargs = kwargs
        self.objective_modules = []
        self.objective_modules_strs = self._kwargs['mpc']['objectives']
        self.addEntry2ParamMap("wu", self._nu)
        self._ineq_modules = ineq_modules


    def set_objectives(self):
        # update paramMap
        module = __import__('robotmpcs')
        self.objective_modules = []
        for class_name in self.objective_modules_strs:
            class_ = getattr(module.models.objectives, class_name)
            self.objective_modules.append(class_(self._ineq_modules, **self._kwargs))
            self._paramMapm, self._npar = self.objective_modules[-1].set_parameters(self._paramMap, self._npar)
        return self._paramMap, self._npar

    def eval_objectives(self, z, p):
        wu = p[self._paramMap["wu"]]
        Wu = diagSX(wu, self._nu)

        J = 0
        for module in self.objective_modules:
            J += module.eval_objective(z,p)
        _, _, qddot, *_ = self.extractVariables(z)
        Ju = ca.dot(qddot, ca.mtimes(Wu, qddot))
        Js = 0
        if self._ns > 0:
            s = z[self._nx]
            ws = p[self._paramMap["ws"]]
            Js += ws * s ** 2
        return J + Ju + Js

    def eval_objectiveN(self, z, p):
        J = self.eval_objectives(z, p)
        return J