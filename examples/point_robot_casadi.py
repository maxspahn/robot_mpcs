import sys
import time
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from urdfenvs.scene_examples.obstacles import wall_obstacles

from robotmpcs.models.casadi_mpc import MPCModelCasadi

class PointRobotCasadi():

    _env: UrdfEnv

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
            'desired_position': [4.0, 0.5, 0.0],
            'epsilon': 0.2,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", content_dict=staticGoalDict)
        obst1Dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 2.9, 0.1], "radius": 1.0},
            "rgba": [0.3, 0.5, 0.6, 1.0],
        }
        sphere_obst_1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphere_obst_1]
        robots = [
            GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        ]
        self._env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True, enforce_real_time=True
        )
        self._sensor = FreeSpaceDecompositionSensor(
                'lidar_sensor_joint',
                nb_rays=40,
                max_radius=10,
                number_constraints=5,
                plotting_interval=1,
                plotting_interval_fsd=1
                
        )
        self._env.reset(pos=np.array([0.0, 0.0, 0.0]))
        self._env.add_goal(self._goal)
        self._env.add_sensor(self._sensor, [0])
        self._env.set_spaces()
        for i in range(len(self._obstacles)):
            self._env.add_obstacle(self._obstacles[i])
        #for wall in wall_obstacles:
        #    self._env.add_obstacle(wall)


    def run(self):
        ob, *_ = self._env.step(np.zeros(3))
        n_steps = 1000
        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            t0 = time.perf_counter()
            arguments = dict(
                    x_0=q,
                    wx=1,
                    wu=0.1,
                    wslack=100,
                    goal=self._goal.position()[0:2],
                    discount=1.1,
                    body_radius=0.4,
            )
            #print("Distance")
            for i in range(self._planner.number_planes()):
                plane = ob['robot_0']['FreeSpaceDecompSensor'][f'constraint_{i}']

                distance = self._planner.dist_to_plane(q[0:2], plane, 0.4)
                #print(distance)
                arguments[f'plane_{i}'] = plane
            action = self._planner.compute_action(**arguments)
            print(action)
            t1 = time.perf_counter()
            print(t1-t0)
            ob, *_ = self._env.step(action)

def main():
    point_robot_example = PointRobotCasadi()
    point_robot_example.run()


if __name__ == "__main__":
    main()
