import sys
import os
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpc_example import MpcExample

class PandaMpcExample(MpcExample):


    def initialize_environment(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        robots = [
            GenericUrdfReacher(urdf="panda.urdf", mode=self._config['mpc']['control_mode']),
        ]
        self._env = gym.make(
            "urdf-env-v0",
             render=self._render,
             dt=self._planner._config.time_step,
             robots=robots)
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
        )

        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": [0.1, -0.6, 0.4],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 5.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": [0.1, 0.0, 0.0],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        self._goal = GoalComposition(name="goal", content_dict=goal_dict)
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
        }
        sphereObst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
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

        self._limits_u = np.array([
                [-1, 1],
                [-1, 1],
                [-15, 15],
                [-15, 15],
                [-7.5, 7.5],
                [-10, 10],
                [-12.5, 12.5],
        ])

        self._goal = GoalComposition(name="goal", content_dict=goal_dict)
        pos0 = np.median(self._limits, axis = 1)
        # vel0 = np.array([0.1, 0.0, 0.0])
        self._env.reset(pos=pos0)
        self._env.add_sensor(full_sensor, [0])
        self._env.add_goal(self._goal.sub_goals()[0])
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.set_spaces()

    def run(self):
        action = np.zeros((7,))
        ob, *_ = self._env.step(action)
        n_steps = 1000
        for i in range(n_steps):
            q = ob["robot_0"]['joint_state']['position']
            qdot = ob["robot_0"]['joint_state']['velocity']
            action, output = self._planner.computeAction(q, qdot)
            ob, *_ = self._env.step(action)

def main():
    robot_type = "panda" #options: boxer, po1ntRobot, panda
    setup_file = 'config/' + str(robot_type) + "Mpc.yaml"
    panda_example = PandaMpcExample(setup_file)
    panda_example.initialize_environment()
    panda_example.set_mpc_parameter()
    panda_example.run()


if __name__ == "__main__":
    main()
