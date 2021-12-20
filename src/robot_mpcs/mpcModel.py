import casadi as ca
import numpy as np
import forcespro


def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a


class MpcModel(object):
    def __init__(self, m, n, N):
        self._m = m
        self._nx = 2 * n
        self._nu = n
        self._ns = 0
        self._n = n
        self._n_obst = 0
        self._m_obst = 0
        self._pairs = []
        self._limits ={
            'x': {'low': np.ones(self._nx) * -100, 'high': np.ones(self._nx) * 100},
            'u': {'low': np.ones(self._nu) * -100, 'high': np.ones(self._nu) * 100},
        }
        self._N = N
        self.initParamMap()

    def initParamMap(self):
        self._paramMap = {}
        self._npar = 0
        self.addEntry2ParamMap('wu', self._n)
        self.addEntry2ParamMap('wvel', self._n)
        self.addEntry2ParamMap('w', self._m)
        if self._ns > 0:
            self.addEntry2ParamMap('ws', 1)
        self.addEntry2ParamMap('g', self._m)
        self.addEntry2ParamMap('r_body', 1)
        self.addEntry2ParamMap('lower_limits', self._n)
        self.addEntry2ParamMap('upper_limits', self._n)

    def addEntry2ParamMap(self, name, n_par):
        self._paramMap[name] = list(range(self._npar, self._npar + n_par))
        self._npar += n_par

    def setSlack(self):
        self._ns = 1

    def setSelfCollisionAvoidance(self, pairs):
        self._pairs = pairs

    def setObstacles(self, n_obst, m_obst):
        self._n_obst = n_obst
        self._m_obst = m_obst
        self.addEntry2ParamMap('obst', (m_obst + 1) * n_obst)

    def eval_objectiveCommon(self, z, p):
        q = z[0:self._n]
        qdot = z[self._n: self._nx]
        w = p[self._paramMap['w']]
        wvel = p[self._paramMap['wvel']]
        g = p[self._paramMap['g']]
        W = diagSX(w, self._m)
        Wvel = diagSX(wvel, self._n)
        fk_ee = self._fk.fk(q, self._n, positionOnly=True)
        err = fk_ee - g
        Jx = ca.dot(err, ca.mtimes(W, err))
        Jvel = ca.dot(qdot, ca.mtimes(Wvel, qdot))
        if self._ns > 0:
            s = z[self._nx]
            ws = p[self._paramMap['ws']]
            J = Jx + Jvel + ws * s**2
        else:
            J = Jx + Jvel
        return J

    def eval_objectiveN(self, z, p):
        Jc = self.eval_objectiveCommon(z, p)
        return Jc

    def eval_objective(self, z, p):
        wu = p[self._paramMap['wu']]
        Wu = diagSX(wu, self._n)
        Jc = self.eval_objectiveCommon(z, p)
        qddot = z[self._nx + self._ns: self._nx + self._ns + self._nu]
        Ju = ca.dot(qddot, ca.mtimes(Wu, qddot))
        return Jc + Ju

    def eval_inequalities(self, z, p):
        q = z[0:self._n]
        qdot = z[self._n:self._nx]
        if self._ns > 0:
            s = z[self._nx]
        else:
            s = 0.0
        qddot = z[self._nx + self._ns : self._ns + self._nx + self._nu]
        ineqs = []
        if 'obst' in self._paramMap.keys():
            obsts = p[self._paramMap["obst"]]
            r_body = p[self._paramMap["r_body"]]
            for j in range(self._n):
                fk = self._fk.fk(q, j + 1, positionOnly=True)
                for i in range(self._n_obst):
                    obst = obsts[i * (self._m_obst + 1) : (i + 1) * (self._m_obst + 1)]
                    x = obst[0:self._m_obst]
                    r = obst[self._m_obst]
                    dist = ca.norm_2(fk - x)
                    ineqs.append(dist - r - r_body + s)
        all_ineqs = ineqs + self.eval_jointLimits(z, p) + self.eval_selfCollision(z, p)
        return all_ineqs

    def eval_selfCollision(self, z, p):
        if self._ns > 0:
            s = z[self._nx]
        else:
            s = 0.0
        q = z[0:self._n]
        r_body = p[self._paramMap["r_body"]]
        ineqs = []
        for pair in self._pairs:
            fk1 = self._fk.fk(q, pair[0], positionOnly=True)
            fk2 = self._fk.fk(q, pair[1], positionOnly=True)
            dist = ca.norm_2(fk1 - fk2)
            ineqs.append(dist - (2 * r_body) + s)
        return ineqs

    def eval_jointLimits(self, z, p):
        # Parameters in state boundaries?
        q = z[0:self._n]
        if self._ns > 0:
            s = z[self._nx]
        else:
            s = 0.0
        lower_limits = p[self._paramMap["lower_limits"]]
        upper_limits = p[self._paramMap["upper_limits"]]
        ineqs = []
        for j in range(self._n):
            dist_lower = q[j] - lower_limits[j]
            dist_upper = upper_limits[j] - q[j]
            ineqs.append(dist_lower + s)
            ineqs.append(dist_upper + s)
        return ineqs

    def setLimits(self, limits):
        self._limits = limits

    def continuous_dynamics(self, x, u):
        qdot = x[self._n:2*self._n]
        qddot = u[self._ns:self._n + self._ns]
        acc = ca.vertcat(qdot, qddot)
        return acc

    def setDt(self, dt):
        self._dt = dt

    def setModel(self):
        self._model = forcespro.nlp.SymbolicModel(self._N)
        self._model.continuous_dynamics = self.continuous_dynamics
        self._model.objective = self.eval_objective
        self._model.objectiveN = self.eval_objectiveN
        E = np.concatenate([np.eye(self._nx), np.zeros((self._nx, self._nu + self._ns))], axis=1)
        self._model.E = E
        self._model.lb = np.concatenate((self._limits['x']['low'], self._limits['u']['low']))
        self._model.ub = np.concatenate((self._limits['x']['high'], self._limits['u']['high']))
        self._model.npar = self._npar
        self._model.nvar = self._nx + self._nu + self._ns
        self._model.neq = self._nx
        nbInequalities = self._n_obst * self._n + 2 * self._n + len(self._pairs)
        self._model.nh = nbInequalities
        self._model.hu = np.ones(nbInequalities) * np.inf
        self._model.hl = np.zeros(nbInequalities)
        self._model.ineq = self.eval_inequalities
        self._model.xinitidx = range(0, self._nx)

    def setCodeoptions(self, solverName, debug=False):
        self._codeoptions = forcespro.CodeOptions(solverName)
        self._codeoptions.nlp.integrator.type = "ERK2"
        self._codeoptions.nlp.integrator.Ts = self._dt
        self._codeoptions.nlp.integrator.nodes = 5
        if debug:
            self._codeoptions.printlevel = 1
            self._codeoptions.optlevel = 0
        else:
            self._codeoptions.printlevel = 0
            self._codeoptions.optlevel = 3

    def generateSolver(self):
        _ = self._model.generate_solver(self._codeoptions)
