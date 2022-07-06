import casadi as ca
import numpy as np
import forcespro
import yaml
from shutil import move
from glob import glob


def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a


class MpcModel(object):
    def __init__(self, m, n, N, initParamMap=True):
        self._m = m
        self._nx = 2 * n
        self._nu = n
        self._ns = 0
        self._n = n
        self._n_obst = 0
        self._m_obst = 0
        self._pairs = []
        self._N = N
        if initParamMap:
            self._limits = {
                "x": {"low": np.ones(self._nx) * -100, "high": np.ones(self._nx) * 100},
                "u": {"low": np.ones(self._nu) * -100, "high": np.ones(self._nu) * 100},
                "s": {"low": np.zeros(1), "high": np.ones(1) * np.inf},
            }
            self.initParamMap()

    def initParamMap(self):
        self._paramMap = {}
        self._npar = 0
        self.addEntry2ParamMap("wu", self._nu)
        self.addEntry2ParamMap("wvel", self._n)
        self.addEntry2ParamMap("w", self._m)
        if self._ns > 0:
            self.addEntry2ParamMap("ws", 1)
        self.addEntry2ParamMap("g", self._m)
        self.addEntry2ParamMap("r_body", 1)
        self.addEntry2ParamMap("lower_limits", self._n)
        self.addEntry2ParamMap("upper_limits", self._n)

    def addEntry2ParamMap(self, name, n_par):
        self._paramMap[name] = list(range(self._npar, self._npar + n_par))
        self._npar += n_par

    def setSlack(self):
        self._ns = 1
        self.addEntry2ParamMap("ws", 1)

    def setSelfCollisionAvoidance(self, pairs):
        self._pairs = pairs

    def setObstacles(self, n_obst, m_obst, inCostFunction=False):
        self._n_obst = n_obst
        self._m_obst = m_obst
        self.addEntry2ParamMap("obst", (m_obst + 1) * n_obst)
        self._obstaclesInCosts = inCostFunction
        self.addEntry2ParamMap('wobst', 1)

    def extractVariables(self, z):
        q = z[0: self._n]
        qdot = z[self._n: self._nx]
        qddot = z[self._nx + self._ns : self._nx + self._ns + self._nu]
        return q, qdot, qddot

    def eval_objectiveCommon(self, z, p):
        q, qdot, *_ = self.extractVariables(z)
        w = p[self._paramMap["w"]]
        wvel = p[self._paramMap["wvel"]]
        g = p[self._paramMap["g"]]
        W = diagSX(w, self._m)
        Wvel = diagSX(wvel, self._nu)
        fk_ee = self._fk.fk(q, self._n, positionOnly=True)[0:self._m]
        Jvel = ca.dot(qdot, ca.mtimes(Wvel, qdot))
        err = fk_ee - g
        Jx = ca.dot(err, ca.mtimes(W, err))
        Jobst = 0
        Js = 0
        if self._obstaclesInCosts:
            obstDistances = 1/ca.vcat(self.eval_obstacleDistances(z, p) )
            wobst = ca.SX(np.ones(obstDistances.shape[0]) * p[self._paramMap['wobst']])
            Wobst = diagSX(wobst, obstDistances.shape[0])
            Jobst += ca.dot(obstDistances, ca.mtimes(Wobst, obstDistances))
        if self._ns > 0:
            s = z[self._nx]
            ws = p[self._paramMap["ws"]]
            Js += ws * s ** 2
        return Jx, Jvel, Js, Jobst

    def eval_objectiveN(self, z, p):
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        return Jx + Jvel + Js + Jobst

    def eval_objective(self, z, p):
        wu = p[self._paramMap["wu"]]
        Wu = diagSX(wu, self._nu)
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        _, _, qddot, *_ = self.extractVariables(z)
        Ju = ca.dot(qddot, ca.mtimes(Wu, qddot))
        return Jx + Jvel + Js + Jobst + Ju

    def eval_inequalities(self, z, p):
        all_ineqs = self.eval_obstacleDistances(z, p) + self.eval_jointLimits(z, p) + self.eval_selfCollision(z, p)
        if self._ns > 0:
            s = z[self._nx]
            for ineq in all_ineqs:
                ineq  += s
        return all_ineqs

    def eval_obstacleDistances(self, z, p):
        ineqs = []
        q, *_ = self.extractVariables(z)
        if self._ns > 0:
            s = z[self._nx]
        else:
            s = 0.0
        if "obst" in self._paramMap.keys():
            obsts = p[self._paramMap["obst"]]
            r_body = p[self._paramMap["r_body"]]
            for j in range(self._n):
                fk = self._fk.fk(q, j + 1, positionOnly=True)[0:self._m]
                for i in range(self._n_obst):
                    obst = obsts[i * (self._m_obst + 1) : (i + 1) * (self._m_obst + 1)]
                    x = obst[0 : self._m_obst]
                    r = obst[self._m_obst]
                    dist = ca.norm_2(fk - x)
                    ineqs.append(dist - r - r_body)
        return ineqs

    def eval_selfCollision(self, z, p):
        q, *_ = self.extractVariables(z)
        r_body = p[self._paramMap["r_body"]]
        ineqs = []
        for pair in self._pairs:
            fk1 = self._fk.fk(q, pair[0], positionOnly=True)[0: self._m]
            fk2 = self._fk.fk(q, pair[1], positionOnly=True)[0: self._m]
            dist = ca.norm_2(fk1 - fk2)
            ineqs.append(dist - (2 * r_body))
        return ineqs

    def eval_jointLimits(self, z, p):
        # Parameters in state boundaries?
        q, *_ = self.extractVariables(z)
        lower_limits = p[self._paramMap["lower_limits"]]
        upper_limits = p[self._paramMap["upper_limits"]]
        ineqs = []
        for j in range(self._n):
            dist_lower = q[j] - lower_limits[j]
            dist_upper = upper_limits[j] - q[j]
            ineqs.append(dist_lower)
            ineqs.append(dist_upper)
        return ineqs

    def setLimits(self, limits):
        self._limits = limits

    def continuous_dynamics(self, x, u):
        qdot = x[self._n: self._nx]
        qddot = u[-self._nu:]
        acc = ca.vertcat(qdot, qddot)
        return acc

    def setDt(self, dt):
        self._dt = dt

    def setModel(self):
        self._model = forcespro.nlp.SymbolicModel(self._N)
        self._model.continuous_dynamics = self.continuous_dynamics
        self._model.objective = self.eval_objective
        self._model.objectiveN = self.eval_objectiveN
        E = np.concatenate(
            [np.eye(self._nx), np.zeros((self._nx, self._nu + self._ns))], axis=1
        )
        self._model.E = E
        if self._ns > 0:
            self._model.lb = np.concatenate(
                (self._limits["x"]["low"], self._limits["s"]["low"], self._limits["u"]["low"])
            )
            self._model.ub = np.concatenate(
                (self._limits["x"]["high"], self._limits["s"]["high"], self._limits["u"]["high"])
            )
        else:
            self._model.lb = np.concatenate(
                (self._limits["x"]["low"], self._limits["u"]["low"])
            )
            self._model.ub = np.concatenate(
                (self._limits["x"]["high"], self._limits["u"]["high"])
            )
        self._model.npar = self._npar
        self._model.nvar = self._nx + self._nu + self._ns
        self._model.neq = self._nx
        nbInequalities = self._n_obst * self._n + 2 * self._n + len(self._pairs)
        self._model.nh = nbInequalities
        self._model.hu = np.ones(nbInequalities) * np.inf
        self._model.hl = np.zeros(nbInequalities)
        self._model.ineq = self.eval_inequalities
        self._model.xinitidx = range(0, self._nx)

    def setCodeoptions(self, **kwargs):
        debug = False
        solverName = self._modelName + "_n" + str(self._n) + "_" + str(self._dt).replace('.','') + "_H" + str(self._N)
        if self._ns == 0:
            solverName += "_noSlack"
        if debug in kwargs:
            debug = kwargs.get('debug')
        if solverName in kwargs:
            solverName = kwargs.get('solverName')
        self._solverName = solverName
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

    def generateSolver(self, location="./"):
        _ = self._model.generate_solver(self._codeoptions)
        with open(self._solverName + '/paramMap.yaml', 'w') as outfile:
            yaml.dump(self._paramMap, outfile, default_flow_style=False)
        properties = {"nx": self._nx, "nu": self._nu, "npar": self._npar, "ns": self._ns, "m": self._m}
        with open(self._solverName + '/properties.yaml', 'w') as outfile:
            yaml.dump(properties, outfile, default_flow_style=False)
        move(self._solverName, location + self._solverName)
        for file in glob(r'*.forces'):
            move(file, location)

