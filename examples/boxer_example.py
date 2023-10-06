import sys
import os
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.sensors.occupancy_sensor import OccupancySensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from mpc_example import MpcExample
from robotmpcs.planner.visualizer import Visualizer

class BoxerMpcExample(MpcExample):

    def initialize_environment(self):
        self._visualizer = Visualizer()

        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 'origin',
                "child_link": 'ee_link',
                "desired_position": [8.2, -0.2],
                "epsilon": 0.4,
                "type": "staticSubGoal"
            }
        }
        self._goal = GoalComposition(name="goal1", content_dict=goal_dict)
        obst1Dict = {
            "type": "sphere",
            "geometry": {"position": [4.0, -0.5, 0.0], "radius": 1.0},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphereObst1]
        self._r_body = 0.6
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        self._limits_vel = np.array([
                [0.7*(-4), 0.7*4],
                [0.5*(-10), 0.5*10],
        ])
        self._limits_u = np.array([
                [-np.inf, np.inf],
                [-np.inf, np.inf],
        ])
        current_path = os.path.dirname(os.path.abspath(__file__))
        robots = [
            GenericDiffDriveRobot(
                urdf=current_path + "/assets/boxer/boxer.urdf",
                mode='acc', #todo add to config
                actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
                castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
                wheel_radius=0.08,
                wheel_distance=0.494,
                spawn_rotation=np.pi / 2,
            ),
        ]
        self._env = gym.make(
            'urdf-env-v0',
            render=self._render,
            robots=robots,
            dt=self._planner._config.time_step
        )
        for i in range(10):
            self._env.add_visualization(size=[0.6, 0.1]) #todo add r used for mpc

    def run(self):
        q0 = np.median(self._limits, axis = 1)
        ob, *_ = self._env.reset(pos=q0)
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)

        # add sensor
        val = 40
        sensor = OccupancySensor(
            limits=np.array([[-5, 5], [-5, 5], [0, 50 / val]]),
            resolution=np.array([val + 1, val + 1, 5], dtype=int),
            interval=100,
            plotting_interval=100,
        )

        self._env.add_sensor(sensor, [0])
        self._env.set_spaces()

        n_steps = 1000
        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            qdot = ob['robot_0']['joint_state']['velocity']
            vel = np.array((ob['robot_0']['joint_state']['forward_velocity'], qdot[2]), dtype=float)
            action,output = self._planner.computeAction(q, qdot, vel)
            plan = []
            for key in output:
                plan.append(np.concatenate([output[key][:2],np.zeros(1)]))
            ob, *_ = self._env.step(action)
            self._env.update_visualizations(plan)

def main():
    boxer_example = BoxerMpcExample(sys.argv[1])
    boxer_example.initialize_environment()
    boxer_example.set_mpc_parameter()
    boxer_example.run()


if __name__ == "__main__":
    main()
