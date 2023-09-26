import sys
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpc_example import MpcExample

class PointRobotMpcExample(MpcExample):


    def initialize_environment(self):
        staticGoalDict = {
            "weight": 1.0,
            "is_primary_goal": True,
            'indices': [0, 1],
            'parent_link': 'world',
            'child_link': 'base_link',
            'desired_position': [8.2, -0.2],
            'epsilon': 0.2,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", content_dict=staticGoalDict)
        obst1Dict = {
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
        robots = [
            GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
        ]
        self._env: UrdfEnv = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True,
        )

    def run(self):
        q0 = np.median(self._limits)
        ob, _ = self._env.reset()
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)
        n_steps = 1000
        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            qdot = ob['robot_0']['joint_state']['velocity']
            action = self._planner.computeAction(q, qdot)
            ob, *_ = self._env.step(action)

def main():
    point_robot_example = PointRobotMpcExample(sys.argv[1])
    point_robot_example.initialize_environment()
    point_robot_example.set_mpc_parameter()
    point_robot_example.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a config file for solver generation.")
    else:
        main()
