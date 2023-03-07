from MotionPlanningEnv.sphereObstacle import dataclass
import casadi as ca
from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import numpy as np
import forcespro
import yaml
from shutil import move
from glob import glob

from robotmpcs.models.mpcModel import MpcModel


def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a


@dataclass
class MpcConfiguration:
    time_horizon: int
    time_step: float
    weights: dict
    slack: bool
    interval: int
    number_obstacles: int
    model_name: str
    n: int


@dataclass
class RobotConfiguration:
    collision_links: list
    selfCollision: dict
    urdf_file: str
    root_link: str
    end_link: str
    base_type: str



class MpcDiffDriveModel(MpcModel):
    def __init__(self, initParamMap=True, **kwargs):
        super().__init__(initParamMap=initParamMap, **kwargs)
        self._n = self._fk.n() + 3
        self._nx = 2 * self._n + 2
        self._nu = 2 + self._fk.n()
        if initParamMap:
            self._limits = {
                "x": {"low": np.ones(self._nx) * -100, "high": np.ones(self._nx) * 100},
                "u": {"low": np.ones(self._nu) * -100, "high": np.ones(self._nu) * 100},
                "s": {"low": np.zeros(1), "high": np.ones(1) * np.inf},
            }
            self.initParamMap()

    def get_velocity(self, z):
        return z[2*self._n: 2*self._n + self._nu]

    def continuous_dynamics(self, x, u):
        x_pos = x[0:3]
        vel = x[6:8]
        xdot = self.computeXdot(x_pos, vel)
        veldot = u
        xddot = ca.SX(np.zeros(3))
        state_dot = ca.vertcat(xdot, xddot, veldot)
        return state_dot

    def computeXdot(self, x, vel):
        assert x.size() == (3, 1)
        assert vel.size() == (2, 1)
        xdot = ca.vertcat(
                ca.cos(x[2]) * vel[0],
                ca.sin(x[2]) * vel[0],
                vel[1],
            )
        return xdot
