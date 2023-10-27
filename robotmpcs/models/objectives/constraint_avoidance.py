import casadi as ca
from robotmpcs.models.mpcBase import MpcBase
from robotmpcs.utils.utils import diagSX

class ConstraintAvoidance(MpcBase):

    def __init__(self, ineq_modules,  **kwargs):
        super().__init__(**kwargs)

        self._n_constr_types = len(kwargs['mpc']['constraints'])
        self._ineq_modules = ineq_modules


    def set_parameters(self, ParamMap, npar):
        self._paramMap = ParamMap
        self._npar = npar

        self.addEntry2ParamMap('wconstr', self._n_constr_types)
        return self._paramMap, self._npar


    def eval_objective(self, z, p):
        w = p[self._paramMap["wconstr"]]
        Jconstr = 0
        for i,ineq_module in enumerate(self._ineq_modules):
            for j in range(self._N):
                ineq = ineq_module.eval_constraint(z, p)
                if len(ineq)>0:
                    Wi = diagSX(w[i], ineq[0].shape[0])
                    Jconstr += Wi * 1 / ineq[0]

        return Jconstr