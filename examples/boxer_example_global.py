import sys
from typing import List, Union, Tuple
import os
import numpy as np
import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.sensors.lidar import Lidar
from urdfenvs.sensors.occupancy_sensor import OccupancySensor
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition
from robotmpcs.global_planner import globalPlanner

from robotmpcs.utils.free_space_decomposition import FreeSpaceDecomposition
from robotmpcs.utils.utils import visualize_constraints_over_N_in_pybullet

from mpc_example import MpcExample


class BoxerMpcExample(MpcExample):

    def __init__(self, config_file_name: str):
        super().__init__(config_file_name)

        self._N = self._config['mpc']['time_horizon']
        self._n_obstacles = self._config['mpc']['number_obstacles']

        self._fsd = FreeSpaceDecomposition(number_constraints=self._n_obstacles, max_radius=5)
        self._plotting_interval = -1


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
                "desired_position": [7.2, -2.2],
                "epsilon": 0.4,
                "type": "staticSubGoal"
            }
        }
        self._goal = GoalComposition(name="goal1", content_dict=goal_dict)
        obstacle_1_dict = {
            "type": "sphere",
            "geometry": {"position": [4.0, -1.5, 0.0], "radius": 1.0},
        }
        obstacle_1 = SphereObstacle(name="sphere_1", content_dict=obstacle_1_dict)
        obstacle_2_dict = {
            "type": "sphere",
            "geometry": {"position": [2.4, -0.7, 0.0], "radius": 0.3},
        }
        obstacle_2 = SphereObstacle(name="sphere_2", content_dict=obstacle_2_dict)
        obstacle_3_dict = {
            "type": "box",
            "geometry": {
                "position": [5.0, 2.3, 0.0],
                "width": 3.0,
                "height": 1.3,
                "length": 1.3,
                "orientation": [0, 0.0, 0.0, 1.0], #ignored in urdfenvs
            },
        }
        obstacle_3 = BoxObstacle(name="box_1", content_dict=obstacle_3_dict)
        self._obstacles = [obstacle_1, obstacle_2, obstacle_3]
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

        self._lin_constr = self._N*[self._n_obstacles * [np.array([1, 0, 0, -100])]]
        current_path = os.path.dirname(os.path.abspath(__file__))

        self._env: UrdfEnv = gym.make(
                'urdf-env-v0',
                render=self._render,
                robots=robots,
                dt=self._planner._config.time_step
            )
        for i in range(self._config['mpc']['time_horizon']):
            self._env.add_visualization(size=[self._r_body, 0.1])

    def compute_point_cloud(self, robot_state: np.ndarray, lidar_obs: np.ndarray) -> np.ndarray:
        """
        Computes point cloud based on raw relative lidar measurements.
        """
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
        return absolute_positions

    def compute_constraints(self, robot_state: np.ndarray, point_cloud: np.ndarray) -> Tuple[List, List]:
        """
        Computes linear constraints given a pointcloud as numpy array.
        The seed point is the robot_state.
        """
        angle = robot_state[2]
        rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
        ])

        position_lidar = np.dot(rot_matrix, np.array([0.4, 0.0])) + robot_state[0:2]
        lidar_position = np.array([position_lidar[0], position_lidar[1], 0.02])

        self._fsd.set_position(lidar_position)
        self._fsd.compute_constraints(point_cloud)
        return list(self._fsd.asdict().values()), self._fsd.constraints()



    def run(self):
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)

        number_lidar_rays = 64
        lidar = Lidar(
                "ee_joint",
                nb_rays=number_lidar_rays,
                raw_data=False,
                plotting_interval=self._plotting_interval,
                angle_limits=np.array([-np.pi + np.pi/8, -np.pi/8]),
        )
        val = 40
        occ_sensor = OccupancySensor(
            limits=np.array([[-5, 10], [-5, 10], [0, 50 / val]]),
            resolution=np.array([val + 1, val + 1, 5], dtype=int),
            interval=100,
            plotting_interval=100,
        )
        self._env.add_sensor(lidar, [0])
        self._env.add_sensor(occ_sensor, [0])
        self._env.set_spaces()

        global_planner = globalPlanner.GlobalPlanner(dim_pixels = occ_sensor._resolution,
                                                     limits_low = occ_sensor._limits.transpose()[0],
                                                     limits_high = occ_sensor._limits.transpose()[1],
                                                     BOOL_PLOTTING=True)
        global_path = []



        ob, *_ = self._env.reset()

        self._env.add_goal(self._goal)
        ob, *_ = self._env.step(np.array([0.0,0.0]))


        n_steps = 1000
        exitflag = 0
        output: Union[dict, None] = None

        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            qdot = ob['robot_0']['joint_state']['velocity']
            vel = np.array((ob['robot_0']['joint_state']['forward_velocity'], qdot[2]), dtype=float)
            lidar_obs = ob['robot_0']['LidarSensor']
            point_cloud = self.compute_point_cloud(q, lidar_obs)

            if i == 0:
                global_planner.get_occupancy_map(occ_sensor, ob['robot_0']['Occupancy'])
                goal_pos = self._goal._config.subgoal0.desired_position + [0]
                global_path, _ = global_planner.get_global_path_astar(start_pos=q, goal_pos=goal_pos)
                global_planner.add_path_to_env(path_length=len(global_path), env=self._env)


            # ---START: Preprocessing for planner---
            # This part is the preprocessing of perception for the planner.
            # Ideally, this would be part of a preprocessor class. There could
            # be one for pointcloud to linear constraints, one for dynamic
            # obstacles, one for static perceived obstacles and so on
            linear_constraints = []
            halfplanes = []
            for j in range(self._N):
                if not output is None and exitflag >= 0:
                    key = "x{:02d}".format(j+1)
                    ref_q = output[key][0:3]
                else:
                    ref_q = q
                linear_constraints_j, halfplanes_j = self.compute_constraints(ref_q, point_cloud)
                linear_constraints.append(linear_constraints_j)
                halfplanes.append(halfplanes_j)

            self._planner.setLinearConstraints(linear_constraints, r_body=self._r_body)
            # ---END: Preprocessing for planner---
            action, output, exitflag = self._planner.computeAction(q, qdot, vel)
            # ---START: Visualizations---
            plan = []
            for key in output:
                plan.append(np.concatenate([output[key][:2],np.zeros(1)]))
            ob, *_ = self._env.step(action)
            if self.check_goal_reaching(ob):
                print("goal reached")
                break
            if self._plotting_interval > 0 and i % self._plotting_interval == 0:
                visualize_constraints_over_N_in_pybullet(halfplanes, 0.02)
            self._env.update_visualizations(plan+global_path)
            # ---END: Visualizations---

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
