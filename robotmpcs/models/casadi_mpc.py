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

    def p_np(self, name: str) -> np.ndarray:
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

    def __init__(self, steps: int, time_step: float):
        self._name = "Casadi_Model"
        self._options = {}
        self._options['ipopt'] = {}
        self._options['print_time'] = 0
        self._options['ipopt']['print_level'] = 0
        self._steps = steps
        self._time_step = time_step
        self._problem = ca.Opti()
        self._initial_x = np.zeros((3, steps+1))
        self._initial_u = np.zeros((3, steps))
        self.init_parameters()
        self.init_problem()

    def init_problem(self):
        self._x = self._problem.variable(3, self._steps + 1)
        self._u = self._problem.variable(3, self._steps)
        self._slack = self._problem.variable(1, self._steps)
        self._problem.minimize(self.objective())
        for k in range(self._steps):
            x_next = self._x[:, k] + self._u[:, k] * self._time_step
            self._problem.subject_to(self._x[:, k+1]==x_next)

        x_0 = self._parameters.p_ca('x_0')
        self._problem.subject_to(self._x[:, 0]==x_0)
        self.set_obstacles()

    def set_obstacles(self):
        o_pos = self._parameters.p_ca('o_pos')
        o_radius = self._parameters.p_ca('o_radius')
        body_radius = self._parameters.p_ca('body_radius')
        for k in range(self._steps):
            distance = ca.norm_2(self._x[0:2, k] - o_pos[0:2]) - o_radius - body_radius
            self._problem.subject_to(distance+self._slack[:, k]>0)


    def init_parameters(self):
        self._parameters = Parameters()
        self._parameters.add(self._problem, 1, 'wu', 1)
        self._parameters.add(self._problem, 1, 'wx', 1)
        self._parameters.add(self._problem, 1, 'wslack', 1)
        self._parameters.add(self._problem, 1, 'goal', 3)
        self._parameters.add(self._problem, 1, 'x_0', 3)
        self._parameters.add(self._problem, 1, 'discount', 1)
        self._parameters.add(self._problem, 1, 'o_pos', 3)
        self._parameters.add(self._problem, 1, 'o_radius', 1)
        self._parameters.add(self._problem, 1, 'body_radius', 1)

    def objective(self) -> ca.SX:
        w_u = self._parameters.p_ca('wu')
        w_x = self._parameters.p_ca('wx')
        w_slack = self._parameters.p_ca('wslack')
        W_x = diagMX(w_x, 3)
        W_u = diagMX(w_u, 3)
        goal = self._parameters.p_ca('goal')
        J_x = 0
        J_u = 0
        J_s = 0
        discount_factor = self._parameters.p_ca('discount')
        for i in range(self._steps):
            err = self._x[0:3, i+1] - goal
            J_x += discount_factor ** i * ca.dot(err, ca.mtimes(W_x, err))
            J_s += w_slack * self._slack[0, i]**2
            J_u += ca.dot(self._u[:, i], ca.mtimes(W_u, self._u[:, i]))
        return J_x + J_u + J_s

    def compute_action(self, **kwargs) -> np.ndarray:
        self._problem.solver('ipopt', self._options)
        self._parameters.set_values(kwargs)
        for name in self._parameters.names():
            symbol = self._parameters.p_ca(name)
            value = self._parameters.p_np(name)
            self._problem.set_value(symbol, value)
        self._problem.set_initial(self._x, self._initial_x)
        self._problem.set_initial(self._u, self._initial_u)
        sol = self._problem.solve()
        #self._initial_x[:, :-1] = sol.value(self._x[:, 1:])
        #self._initial_u[:, :-1] = sol.value(self._u[:, 1:])
        return sol.value(self._u[:, 0])
