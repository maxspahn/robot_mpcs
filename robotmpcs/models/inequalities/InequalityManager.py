from robotmpcs.models.mpcBase import MpcBase

class InequalityManager(MpcBase):

    def __init__(self, ParamMap={}, npar=0, **kwargs):
        super().__init__(**kwargs)

        self._paramMap = ParamMap
        self._npar = npar
        self._kwargs = kwargs
        self.inequality_modules = []
        self.inequality_modules_strs = self._kwargs['mpc']['constraints']


    def set_constraints(self):
        # update paramMap
        module = __import__('robotmpcs')
        self.inequality_modules = []
        for class_name in self.inequality_modules_strs:
            class_ = getattr(module.models.inequalities, class_name)
            self.inequality_modules.append(class_(**self._kwargs))
            self._paramMapm, self._npar = self.inequality_modules[-1].set_parameters(self._paramMap, self._npar)
        return self._paramMap, self._npar

    def eval_inequalities(self, z, p):
        all_ineqs = []
        for module in self.inequality_modules:
            all_ineqs += module.eval_constraint(z,p)
        if self._ns > 0:
            slack_variable = z[self._nx]
            for ineq in all_ineqs:
                ineq += slack_variable
        return all_ineqs



