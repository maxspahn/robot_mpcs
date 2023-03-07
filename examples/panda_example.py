import sys
import numpy as np
import gym
import urdfenvs.panda_reacher
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from mpc_example import MpcExample

class PandaMpcExample(MpcExample):


    def initialize_environment(self):
        staticGoalDict = {
            "m": 3,
            "w": 1.0,
            "prime": True,
            'indices': [0, 1, 2],
            'parent_link': 'panda_link0',
            'child_link': 'panda_link7',
            'desired_position': [-0.3, -0.4, 0.2],
            'epsilon': 0.02,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
        obst1Dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [0.1, -0.3, 0.3], "radius": 0.15},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", contentDict=obst1Dict)
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
        self._env = gym.make(
            'panda-reacher-acc-v0',
             render=self._render,
             dt=self._planner._config.time_step)

    def run(self):
        q0 = np.median(self._limits)
        ob = self._env.reset(pos=q0)
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)
        n_steps = 1000
        for i in range(n_steps):
            q = ob['joint_state']['position']
            qdot = ob['joint_state']['velocity']
            action = self._planner.computeAction(q, qdot)
            ob, *_ = self._env.step(action)

def main():
    panda_example = PandaMpcExample(sys.argv[1])
    panda_example.initialize_environment()
    panda_example.set_mpc_parameter()
    panda_example.run()


if __name__ == "__main__":
    main()
