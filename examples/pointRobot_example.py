import sys
import numpy as np
import gym
import urdfenvs.point_robot_urdf
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from mpc_example import MpcExample

class PointRobotMpcExample(MpcExample):


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
        self._r_body = 0.3
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        self._env = gym.make(
            'pointRobotUrdf-acc-v0',
            render=self._render,
            dt=self._planner._config.time_step,
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
