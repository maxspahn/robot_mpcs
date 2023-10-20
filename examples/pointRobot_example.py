import sys
import os
import numpy as np
import gymnasium as gym
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from mpc_example import MpcExample
from robotmpcs.planner.visualizer import Visualizer

class PointRobotMpcExample(MpcExample):

    def initialize_environment(self):
        self._visualizer = Visualizer()

        robots = [
            GenericUrdfReacher(urdf="pointRobot.urdf", mode=self._config['mpc']['control_mode']),
        ]

        self._env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )
        # Set the initial position and velocity of the point mass.
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
        )
        self._r_body = 0.3
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        self._limits_u = np.array([
                [-1, 1],
                [-1, 1],
                [-15, 15],
        ])
        # Definition of the obstacle.
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [4.0, -0.5, 0.0], "radius": 1.0},
        }
        obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
        self._obstacles = [obst1]
        # Definition of the goal.
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
        self._goal = GoalComposition(name="goal", content_dict=goal_dict)
        pos0 = np.median(self._limits, axis = 1)
        vel0 = np.array([0.1, 0.0, 0.0])
        self._env.reset(pos=pos0, vel=vel0)
        self._env.add_sensor(full_sensor, [0])
        self._env.add_goal(self._goal.sub_goals()[0])
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.set_spaces()

        for i in range(self._config['mpc']['time_horizon']):
            self._env.add_visualization(size=[self._r_body, 0.1])

        return {}

    def run(self):
        action = np.array([0.0, 0.0, 0.0])
        ob, *_ = self._env.step(action)
        n_steps = 1000
        for i in range(n_steps):
            q = ob["robot_0"]['joint_state']['position']
            qdot = ob["robot_0"]['joint_state']['velocity']
            action, output = self._planner.computeAction(q, qdot)
            plan = []
            for key in output:
                plan.append(np.concatenate([output[key][:2],np.zeros(1)]))
            ob, *_ = self._env.step(action)
            self._env.update_visualizations(plan)

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
