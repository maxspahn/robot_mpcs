import os
import re
import sys
from typing import List
import numpy as np
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.goal_composition import GoalComposition
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from robotmpcs.utils.utils import parse_setup

from robotmpcs.planner.mpcPlanner import MPCPlanner

from robotmpcs.models.diff_drive_mpc_model import MpcDiffDriveModel
from robotmpcs.models.mpcModel import MpcModel


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

        self._config['robot']['urdf_file'] = os.path.dirname(os.path.abspath(__file__)) + "/assets/" + str(self._robot_type) + "/" + \
                                      self._config['robot']['urdf_file']
        if self._config['robot']['base_type'] == 'holonomic':
            mpc_model = MpcModel(initParamMap=True, **self._config)
        elif self._config['robot']['base_type'] == 'diffdrive':
            mpc_model = MpcDiffDriveModel(initParamMap=True, **self._config)
        mpc_model.setModel()
        mpc_model.setCodeoptions()

        self._planner = MPCPlanner(
            self._robot_type,
            self._solver_directory,
            mpc_model,
            self._config['example']['debug'],
            **self._config['mpc'])
        self._planner.concretize()
        self._planner.reset()
        self._n = self._planner._config.n
        self._render = True

    def set_mpc_parameter(self):
        constraints = self._config['mpc']['constraints']
        objectives = self._config['mpc']['objectives']

        for objective in objectives:
            if objective == 'GoalReaching':
                try:
                    self._planner.setGoalReaching(self._goal.primary_goal().position())
                except AttributeError:
                    print('The required attributes for setting ' + objective + ' are not defined')
                    sys.exit(1)
            elif objective == 'ConstraintAvoidance':
                try:
                    self._planner.setConstraintAvoidance()
                except KeyError:
                    print('The required attributes for setting ' + objective + ' are not defined in the config file')
                    sys.exit(1)
            else:
                print('No function to set the parameters for this objective is defined')
                sys.exit(1)

        for constraint in constraints:
            if constraint == 'JointLimitConstraints':
                try:
                    self._planner.setJointLimits(np.transpose(self._limits))
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            elif constraint == 'VelLimitConstraints':
                try:
                    self._planner.setVelLimits(np.transpose(self._limits_vel))
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            elif constraint == 'InputLimitConstraints':
                try: self._planner.setInputLimits(np.transpose(self._limits_u))
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            elif constraint == 'LinearConstraints':
                try: self._planner.setLinearConstraints(self._lin_constr, self._r_body)
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            elif constraint == 'RadialConstraints':
                try: self._planner.setRadialConstraints(self._obstacles, self._r_body)
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            elif constraint == 'SelfCollisionAvoidanceConstraints':
                try: self._planner.setSelfCollisionAvoidanceConstraints(self._r_body)
                except AttributeError:
                    print('The required attributes for setting ' + constraint + ' are not defined')
                    sys.exit(1)
            else:
                print('No function to set the parameters for this constraint type is defined')
                sys.exit(1)


