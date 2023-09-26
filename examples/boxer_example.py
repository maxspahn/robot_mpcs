import sys
import numpy as np
import gymnasium as gym
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from mpc_example import MpcExample

class BoxerMpcExample(MpcExample):



    def initialize_environment(self):
        staticGoalDict = {
            "weight": 1.0,
            "is_primary_goal": True,
            'indices': [0, 1],
            'parent_link': 0,
            'child_link': self._n,
            'desired_position': [2, -2],
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
        self._r_body = 0.6
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        robots = [
            GenericDiffDriveRobot(
                urdf="boxer.urdf",
                mode="acc",
                actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
                castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
                wheel_radius = 0.08,
                wheel_distance = 0.494,
                spawn_rotation=np.pi/2,
            ),
        ]
        self._env = gym.make(
            "urdf-env-v0",
            dt=0.05, robots=robots, render=True
        )

    def run(self):
        with open('boxer_fk.urdf', 'r') as f:
            urdf = f.read()
        fk = GenericURDFFk(
            urdf,
            'base_link',
            'ee_link',
            base_type='diffdrive',
        )
        ob, _ = self._env.reset()
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.add_goal(self._goal)
        n_steps = 10
        for i in range(n_steps):
            q = ob['robot_0']['joint_state']['position']
            qdot = ob['robot_0']['joint_state']['velocity']
            #q[2] = q[2] - np.pi/2
            vel = np.array((ob['robot_0']['joint_state']['forward_velocity'], qdot[2]), dtype=float)
            print(q)
            #print(fk.fk(q, 'base_link', 'ee_link', positionOnly=True))
            action = self._planner.computeAction(q, qdot, vel)
            ob, *_ = self._env.step(action)

def main():
    boxer_example = BoxerMpcExample(sys.argv[1])
    boxer_example.initialize_environment()
    boxer_example.set_mpc_parameter()
    boxer_example.run()


if __name__ == "__main__":
    main()
