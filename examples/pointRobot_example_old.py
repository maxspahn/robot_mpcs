import sys
import os
import numpy as np
import gymnasium as gym
# import urdfenvs.point_robot_urdf
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from mpc_example import MpcExample

class PointRobotMpcExample(MpcExample):


    def initialize_environment(self):
        # staticGoalDict = {
        #     "subgoal0":{
        #     "m": 2,
        #     "w": 1.0,
        #     "prime": True,
        #     'indices': [0, 1],
        #     'parent_link': 0,
        #     'child_link': self._n,
        #     'desired_position': [8.2, -0.2],
        #     'epsilon': 0.2,
        #     'type': "staticSubGoal",
        #     }
        # }
        goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": self._n,
                "desired_position": [8.2, -0.2],
                "epsilon": 0.1,
                "type": "staticSubGoal"
            }
        }
        self._goal = GoalComposition(name="goal1", content_dict=goal_dict)
        obst1Dict = {
            # "dim": 3,
            "type": "sphere",
            "geometry": {"position": [4.0, -0.5, 0.0], "radius": 1.0},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphereObst1]
        self._r_body = 0.3
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        current_path = os.path.dirname(os.path.abspath(__file__))
        robots = [
            GenericUrdfReacher(
                urdf=current_path + "/assets/po1ntRobot/pointRobot.urdf",
                mode="acc",
            )
        ]
        self._env = gym.make(
            'urdf-env-v0',
            robots=robots,
            render=self._render,
            dt=self._planner._config.time_step,
        )

    def run(self):
        # full_sensor = FullSensor(
        #     goal_mask=["position", "weight"],
        #     obstacle_mask=["position", "size"],
        #     variance=0.0,
        # )
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
        )
        q0 = np.median(self._limits, axis = 1)
        ob = self._env.reset(pos=q0)
        self._env.add_sensor(full_sensor, [0])
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)
        # q0 = np.median(self._limits, axis = 1)
        # ob = self._env.reset(pos=q0)
        # # self._env.add_sensor(full_sensor, [0])
        # for obstacle in self._obstacles:
        #     self._env.add_obstacle(obstacle)
        # self._env.add_goal(self._goal)
        n_steps = 1000
        for i in range(n_steps):
            q = ob['joint_state']['position']
            qdot = ob['joint_state']['velocity']
            action = self._planner.computeAction(q, qdot)
            ob, *_ = self._env.step(action)

def main():
    point_robot_example = PointRobotMpcExample(sys.argv[1])
    point_robot_example.initialize_environment()
    # point_robot_example.set_mpc_parameter()
    point_robot_example.run()

if __name__ == "__main__":
    sys.argv.append('config/po1ntRobotMpc.yaml')
    if len(sys.argv) < 2:
        print("Please provide a config file for solver generation.")
    else:
        main()
