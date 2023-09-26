import sys
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpc_example import MpcExample

class PandaMpcExample(MpcExample):


    def initialize_environment(self):
        staticGoalDict = {
            "weight": 1.0,
            "is_primary_goal": True,
            'indices': [0, 1, 2],
            'parent_link': 'panda_link0',
            'child_link': 'panda_link7',
            'desired_position': [-0.3, -0.4, 0.2],
            'epsilon': 0.02,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", content_dict=staticGoalDict)
        obst1Dict = {
            "type": "sphere",
            "geometry": {"position": [0.1, -0.3, 0.3], "radius": 0.15},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphereObst1]
        self._r_body = 0.14
        self._limits = np.array([
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
        ])
        robots = [
            GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
        ]
        self._env: UrdfEnv = gym.make(
            "urdf-env-v0",
            dt=self._planner._config.time_step, robots=robots, render=True,
        )

    def run(self):
        ob, *_ = self._env.reset()
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
    panda_example = PandaMpcExample(sys.argv[1])
    panda_example.initialize_environment()
    panda_example.set_mpc_parameter()
    panda_example.run()


if __name__ == "__main__":
    main()
