import sys
import numpy as np
import gym
import urdfenvs.boxer_robot
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from mpc_example import MpcExample

class BoxerMpcExample(MpcExample):


    def initialize_environment(self):
        staticGoalDict = {
            "m": 2,
            "w": 1.0,
            "prime": True,
            'indices': [0, 1],
            'parent_link': 0,
            'child_link': self._n,
            'desired_position': [8.2, -0.2],
            'epsilon': 0.2,
            'type': "staticSubGoal", 
        }
        self._goal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
        obst1Dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [4.0, -0.5, 0.0], "radius": 1.0},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", contentDict=obst1Dict)
        self._obstacles = [sphereObst1]
        self._r_body = 0.6
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        self._env = gym.make(
            'boxer-robot-acc-v0',
            render=self._render,
            dt=self._planner._config.time_step
        )

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
            vel = np.array((ob['joint_state']['forward_velocity'], qdot[2]), dtype=float)
            action = self._planner.computeAction(q, qdot, vel)
            ob, *_ = self._env.step(action)

def main():
    boxer_example = BoxerMpcExample(sys.argv[1])
    boxer_example.initialize_environment()
    boxer_example.set_mpc_parameter()
    boxer_example.run()


if __name__ == "__main__":
    main()
