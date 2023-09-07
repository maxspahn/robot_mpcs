from typing import Dict, List
import casadi as ca
import numpy as np

def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val
    return a

def diagMX(val, size):
    a = ca.MX(size, size)
    for i in range(size):
        a[i, i] = val
    return a

class Parameters():
    _data_values : Dict[str, np.ndarray]
    _data_symbols : Dict[str, ca.SX]

    def __init__(self):
        self._data_values = {}
        self._data_symbols = {}

    def names(self) -> List[str]:
        return list(self._data_symbols.keys())

    def add(self, problem: ca.Opti, n_steps: int, name: str, dimension: int):
        # self._data_symbols[name] = ca.MX.sym(name, dimension)
        self._data_symbols[name] =  problem.parameter(dimension, n_steps)

    def set_values(self, pairs):
        self._data_values = dict(sorted(pairs.items()))

    def p_ca(self, name: str) -> ca.SX:
        return self._data_symbols[name]

    def p_np(self, name: str) -> ca.SX:
        return self._data_values[name]
    
    def all_p(self) -> ca.SX:
        self._data_symbols = dict(sorted(self._data_symbols.items()))
        values = []
        for parameter in self._data_symbols:
            values.append(self._data_symbols[parameter])
        return ca.vcat(values)

    def all_p_np(self) -> np.ndarray:
        values = []
        for parameter, value in self._data_values.items():
            if isinstance(value, float) or isinstance(value, int):
                values.append(value)
            elif isinstance(value, list):
                values += value
            elif isinstance(value, dict):
                values.append(value.tolist())
        return np.array(values)


class MPCModelCasadi():
    _name : str
    _solver : ca.Function
    _options: dict
    _x: ca.SX
    _parameters: Parameters

    def __init__(self):
        self._name = "Casadi_Model"
        self._options = {}
        self._options['ipopt'] = {}
        self._options['print_time'] = 0
        self._options['ipopt']['print_level'] = 1
        self._steps = 10
        self._time_step = 0.01
        self._problem = ca.Opti()
        self.init_parameters()
        self.init_problem()

    def init_problem(self):
        self._x = self._problem.variable(3, self._steps + 1)
        self._u = self._problem.variable(3, self._steps)
        self._problem.minimize(self.objective())
        for k in range(self._steps):
            x_next = self._x[:, k] + self._u[:, k]
            self._problem.subject_to(self._x[:, k+1]==x_next)


    def init_parameters(self):
        self._parameters = Parameters()
        self._parameters.add(self._problem, 1, 'wu', 1)
        self._parameters.add(self._problem, 1, 'wx', 1)
        self._parameters.add(self._problem, 1, 'goal', 3)

    def objective(self) -> ca.SX:
        wu = self._parameters.p_ca('wu')
        wx = self._parameters.p_ca('wx')
        Wx = diagMX(wx, 3)
        goal = self._parameters.p_ca('goal')
        err = self._x[0:3] - goal
        Jx = ca.dot(err, ca.mtimes(Wx, err))
        return Jx

    def equality_constraints(self) -> ca.SX:
        return self._x[2]+(1-self._x[0])**2-self._x[1]

    def compose_problem(self):
        nlp = dict(
            x=self._x,
            p=self._parameters.all_p(),
            f=self.objective(),
            g=self.equality_constraints(),
        )
        self._solver = ca.nlpsol('solver', 'ipopt', nlp, self._options)

    def lower_bounds_state(self) -> np.ndarray:
        return np.ones(3) * 3

    def upper_bounds_state(self) -> np.ndarray:
        return np.ones(3) * 10

    def lower_bounds_constraints(self) -> np.ndarray:
        return [-np.inf] * 1

    def upper_bounds_constraints(self) -> np.ndarray:
        return [np.inf] * 1


    def solve(self, x_0: np.ndarray, **kwargs) -> np.ndarray:
        self._parameters.set_values(kwargs)
        res = self._solver(
            x0=x_0,
            p=self._parameters.all_p_np(),
            lbx=self.lower_bounds_state(),
            ubx=self.upper_bounds_state(),
            lbg=self.lower_bounds_constraints(),
            ubg=self.upper_bounds_constraints(),
        )
        return res['x']

    def solve_problem(self, **kwargs) -> np.ndarray:
        self._problem.solver('ipopt')
        self._parameters.set_values(kwargs)
        wx_val = ca.MX(1, 1)
        wx_val[0, 0] = 1.0
        for name in self._parameters.names():
            symbol = self._parameters.p_ca(name)
            value = self._parameters.p_np(name)
            self._problem.set_value(symbol, value)
        sol = self._problem.solve()
        return sol.value(self._u[:, 0])

def main():
    model = MPCModelCasadi()
    #model.compose_problem()
    #sol = model.solve(np.array([1.0, 0.3, 0.3]), weight = 0.1, exp =  1)
    goal = np.array([1.0, 1.0, 0.0])
    sol = model.solve_problem(wx=1, wu=1, goal=goal)
    print(sol)


if __name__ == "__main__":
    main()
