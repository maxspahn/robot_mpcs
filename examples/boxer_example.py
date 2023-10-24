import sys
from typing import List
import os
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.sensors.lidar import Lidar
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpc_example import MpcExample
from robotmpcs.models.utils.free_space_decomposition import FreeSpaceDecomposition
from robotmpcs.models.utils.utils import visualize_constraints_in_pybullet


class BoxerMpcExample(MpcExample):

    def __init__(self, config_file_name: str):
        super().__init__(config_file_name)
        self._fsd = FreeSpaceDecomposition(number_constraints=5, max_radius=5)

    def initialize_environment(self):

        robots = [
            GenericDiffDriveRobot(
                urdf='boxer.urdf',
                mode=self._config['mpc']['control_mode'],
                actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
                castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
                wheel_radius=0.08,
                wheel_distance=0.494,
                spawn_rotation=np.pi / 2,
            ),
        ]

        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 'origin',
                "child_link": 'ee_link',
                "desired_position": [6.2, -3.2],
                "epsilon": 0.4,
                "type": "staticSubGoal"
            }
        }
        self._goal = GoalComposition(name="goal1", content_dict=goal_dict)
        obst1Dict = {
            "type": "sphere",
            "geometry": {"position": [4.0, -1.5, 0.0], "radius": 1.0},
        }
        sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        self._obstacles = [sphereObst1]
        self._r_body = 0.6
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])

        self._limits_u = np.array([
                [-10, 10],
                [-10, 10],
        ])

        self._lin_constr = [np.array([1, 0, 0, -1.5])]
        current_path = os.path.dirname(os.path.abspath(__file__))

        self._env: UrdfEnv = gym.make(
                'urdf-env-v0',
                render=self._render,
                robots=robots,
                dt=self._planner._config.time_step
            )
        for i in range(self._config['mpc']['time_horizon']):
            self._env.add_visualization(size=[self._r_body, 0.1])

    def compute_constraints(self, robot_state: np.ndarray, lidar_obs: np.ndarray) -> List:
        angle = robot_state[2]
        rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)], 
                [np.sin(angle), np.cos(angle)],
        ])

        position_lidar = np.dot(rot_matrix, np.array([0.4, 0.0])) + robot_state[0:2]
        number_rays = lidar_obs.shape[0]//2
        lidar_observation = lidar_obs.reshape((number_rays, 2))
        lidar_position = np.array([position_lidar[0], position_lidar[1], 0.02])
        relative_positions = np.concatenate(
            (
                np.reshape(lidar_observation, (number_rays, 2)),
                np.zeros((number_rays, 1)),
            ),
            axis=1,
        )
        absolute_positions = relative_positions + np.repeat(
            lidar_position[np.newaxis, :], number_rays, axis=0
        )

        self._fsd.set_position(lidar_position)
        self._fsd.compute_constraints(absolute_positions)
        return list(self._fsd.asdict().values())



    def run(self):
        # add sensor
        number_lidar_rays = 64
        lidar = Lidar(
                "ee_joint",
                nb_rays=number_lidar_rays,
                raw_data=False,
                plotting_interval=1,
                angle_limits=np.array([-np.pi + np.pi/8, -np.pi/8]),
        )
        self._env.add_sensor(lidar, [0])
        self._env.set_spaces()

        q0 = np.median(self._limits, axis = 1)
        ob, *_ = self._env.reset(pos=q0)
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)

        n_steps = 1000
        for _ in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            qdot = ob['robot_0']['joint_state']['velocity']
            vel = np.array((ob['robot_0']['joint_state']['forward_velocity'], qdot[2]), dtype=float)
            lidar_obs = ob['robot_0']['LidarSensor']

            linear_constraints = self.compute_constraints(q, lidar_obs)

            self._planner.setLinearConstraints(linear_constraints, r_body=self._r_body)
            action,output = self._planner.computeAction(q, qdot, vel)
            plan = []
            for key in output:
                plan.append(np.concatenate([output[key][:2],np.zeros(1)]))
            ob, *_ = self._env.step(action)
            if self.check_goal_reaching(ob):
                print("goal reached")
                break
            visualize_constraints_in_pybullet(self._fsd, 0.02)
            self._env.update_visualizations(plan)

    def check_goal_reaching(self, ob):
        primary_goal = self._goal.primary_goal()
        goal_dist = np.linalg.norm(ob['robot_0']['joint_state']['position'][:2] - primary_goal.position()) # todo remove hard coded dimension, replace it with fk instead
        if goal_dist <= primary_goal.epsilon():
            return True
        return False

def main():
    boxer_example = BoxerMpcExample(sys.argv[1])
    boxer_example.initialize_environment()
    boxer_example.set_mpc_parameter()
    boxer_example.run()


if __name__ == "__main__":
    main()
