import os
import re

import numpy as np
from robotmpcs.planner.mpcPlanner import MPCPlanner, SolverDoesNotExistError

envMap = {
    'planarArm': 'nLink-reacher-acc-v0', 
    'diffDrive': 'ground-robot-acc-v0', 
    'po1ntRobot': 'point-robot-acc-v0', 
    'boxer': 'boxer-robot-acc-v0',
    'panda': 'panda-reacher-acc-v0',
}

class MpcExample(object):
    def __init__(self, config_file_name: str):
        test_setup = os.path.dirname(os.path.realpath(__file__)) + "/" + config_file_name
        self._robot_type = re.findall('\/(\S*)M', config_file_name)[0]
        self._solver_directory = os.path.dirname(os.path.realpath(__file__)) + "/solvers/"
        self._env_name = envMap[self._robot_type]
        try:
            self._planner = MPCPlanner(test_setup, self._robot_type, self._solver_directory)
        except SolverDoesNotExistError as e:
            print(e)
            print("Consider creating it with makeSolver.py")
            return
        self._planner.concretize()
        self._planner.reset()
        self._n = self._planner.n()
        self._render = True

    def set_mpc_parameter(self):
        self._planner.setObstacles(self._obstacles, self._r_body)
        self._planner.setGoal(self._goal)
        self._planner.setJointLimits(np.transpose(self._limits))

