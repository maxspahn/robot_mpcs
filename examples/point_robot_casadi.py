import sys
import time
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from robotmpcs.models.casadi_mpc import MPCModelCasadi

class PointRobotCasadi():

    def __init__(self):
        self.initialize_environment()
        self.set_planner()

    def set_planner(self):
        self._planner = MPCModelCasadi(12, 0.1)


    def initialize_environment(self):
        staticGoalDict = {
            "weight": 1.0,
            "is_primary_goal": True,
            'indices': [0, 1, 2],
            'parent_link': 'world',
            'child_link': 'base_link',
            'desired_position': [4.0, 0.1, 0.0],
            'epsilon': 0.2,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", content_dict=staticGoalDict)
        obst1Dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, -0.1, 0.0], "radius": 1.0},
            "rgba": [0.3, 0.5, 0.6, 1.0],
        }
        sphere_obst_1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphere_obst_1]
        robots = [
            GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        ]
        self._env: UrdfEnv = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True, enforce_real_time=True
        )

    def run(self):
        ob, _ = self._env.reset()
        self._env.add_goal(self._goal)
        for i in range(len(self._obstacles)):
            self._env.add_obstacle(self._obstacles[i])
        n_steps = 1000
        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            t0 = time.perf_counter()
            action = self._planner.compute_action(
                    x_0=q,
                    wx=1,
                    wu=0.1,
                    wslack=1000,
                    goal=self._goal.position(),
                    discount=1.1,
                    o_pos=self._obstacles[0].position(),
                    o_radius=self._obstacles[0].size(),
                    body_radius=0.4,
            )
            t1 = time.perf_counter()
            print(t1-t0)
            ob, *_ = self._env.step(action)

def main():
    point_robot_example = PointRobotCasadi()
    point_robot_example.run()


if __name__ == "__main__":
    main()
