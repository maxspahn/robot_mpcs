from typing import Dict, List
import casadi as ca
from dataclasses import dataclass
from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

@dataclass
class MpcConfiguration:
    time_horizon: int
    time_step: float
    weights: dict
    slack: bool
    interval: int
    constraints: list
    objectives: list
    number_obstacles: int
    model_name: str
    initialization: str
    n: int
    control_mode: str
    name: str = 'mpc'
    debug: bool = False

@dataclass
class RobotConfiguration:
    collision_links: list
    selfCollision: dict
    urdf_file: str
    root_link: str
    end_link: str
    base_type: str
class MpcBase(object):
    _npar: int
    _N: int
    _pairs: List[int]
    _paramMap: Dict[str,List[int]]
    _modelName: str

    def __init__(self, **kwargs):
        self._config = MpcConfiguration(**kwargs['mpc'])
        self._debug = kwargs['example']['debug']
        self._robot_config = RobotConfiguration(**kwargs['robot'])
        with open(self._robot_config.urdf_file, 'r') as f:
            urdf = f.read()
        self._modelName = self._config.model_name
        self._fk = GenericURDFFk(
            urdf,
            self._robot_config.root_link,
            self._robot_config.end_link,
            base_type=self._robot_config.base_type,
        )
        self._m = 3
        self._dt = self._config.time_step
        if self._robot_config.base_type == 'holonomic':
            self._n = self._fk.n()
            self._nx = 2 * self._n
            self._nu = self._n
        elif self._robot_config.base_type == 'diffdrive':
            self._n = self._fk.n() + 3
            self._nx = 2 * self._n + 2
            self._nu = 2 + self._fk.n()
        self._ns = 0
        self._n_obst = 0
        self._m_obst = 3
        self._pairs = []
        self._N = self._config.time_horizon

    def addEntry2ParamMap(self, name, n_par):
        if name not in self._paramMap:
            self._paramMap[name] = list(range(self._npar, self._npar + n_par))
            self._npar += n_par

    def get_velocity(self, z):
        return  z[self._n: self._nx]

    def extractVariables(self, z):
        q = z[0: self._n]
        qdot = z[self._n: self._nx]
        qddot = z[self._nx + self._ns : self._nx + self._ns + self._nu]
        return q, qdot, qddot

    def eval_obstacleDistances(self, z, p):
        ineqs = []
        q, *_ = self.extractVariables(z)
        if "obst" in self._paramMap.keys():
            obsts = p[self._paramMap["obst"]]
            r_body = p[self._paramMap["r_body"]]
            for j, collision_link in enumerate(self._robot_config.collision_links):
                fk = self._fk.fk(
                    q,
                    self._robot_config.root_link,
                    collision_link,
                    positionOnly=True
                )[0:self._m]
                for i in range(self._config.number_obstacles):
                    obst = obsts[i * (self._m_obst + 1) : (i + 1) * (self._m_obst + 1)]
                    x = obst[0 : self._m_obst]
                    r = obst[self._m_obst]
                    dist = ca.norm_2(fk - x)
                    ineqs.append(dist - r - r_body)
        return ineqs

