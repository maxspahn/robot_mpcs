import os
import re
from typing import List
import numpy as np
from utils import parse_setup
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.goal_composition import GoalComposition
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from robotmpcs.planner.mpcPlanner import MPCPlanner


envMap = {
    'planarArm': 'nLink-reacher-acc-v0', 
    'diffDrive': 'ground-robot-acc-v0', 
    'pointRobot': 'point-robot-acc-v0',
    'boxer': 'boxer-robot-acc-v0',
    'panda': 'panda-reacher-acc-v0',
}

class MpcExample(object):
    _obstacles: List[CollisionObstacle]
    _goal: GoalComposition
    _r_body: float
    _limits: np.ndarray
    _limits_u: np.ndarray
    _limits_vel: np.ndarray
    _env: UrdfEnv

    def __init__(self, config_file_name: str):
        test_setup = os.path.dirname(os.path.realpath(__file__)) + "/" + config_file_name

        self._robot_type = re.search(r'/([^/]+)Mpc\.yaml', config_file_name).group(1)
        self._solver_directory = os.path.dirname(os.path.realpath(__file__)) + "/solvers/"
        self._env_name = envMap[self._robot_type]
        self._config = parse_setup(test_setup)
        self._planner = MPCPlanner(
            self._robot_type,
            self._solver_directory,
            **self._config['mpc'])
        self._planner.concretize()
        self._planner.reset()
        self._n = self._planner._config.n
        self._render = True

    def set_mpc_parameter(self):
        self._planner.setObstacles(self._obstacles, self._r_body)
        self._planner.setGoal(self._goal)
        if hasattr(self, '_limits'):#todo also check if they were included in solver
            self._planner.setJointLimits(np.transpose(self._limits))
        if hasattr(self, '_limits_vel'):
            self._planner.setVelLimits(np.transpose(self._limits_vel))
        if hasattr(self, '_limits_u'):
            self._planner.setInputLimits(np.transpose(self._limits_u))

