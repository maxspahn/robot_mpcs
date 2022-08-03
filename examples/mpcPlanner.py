import numpy as np
import yaml
import os
import sys
import forcespro
import gym
import planarenvs.ground_robots
import planarenvs.point_robot
import planarenvs.n_link_reacher
import urdfenvs.boxer_robot
import re
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
import robotmpcs
from robotmpcs.planner.mpcPlanner import MPCPlanner, SolverDoesNotExistError



path_name = os.path.dirname(os.path.realpath(__file__)) + '/'
envMap = {
    'planarArm': 'nLink-reacher-acc-v0', 
    'diffDrive': 'ground-robot-acc-v0', 
    'po1ntRobot': 'point-robot-acc-v0', 
    'boxer': 'boxer-robot-acc-v0',
}
obst1Dict = {
    "dim": 2,
    "type": "sphere",
    "geometry": {"position": [4.0, -0.5], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", contentDict=obst1Dict)

def main():
    test_setup = os.path.dirname(os.path.realpath(__file__)) + "/" + sys.argv[1]
    robotType = re.findall('\/(\S*)M', sys.argv[1])[0]
    solversDir = os.path.dirname(os.path.realpath(__file__)) + "/solvers/"
    envName = envMap[robotType]
    try:
        myMPCPlanner = MPCPlanner(test_setup, robotType, solversDir)
    except SolverDoesNotExistError as e:
        print(e)
        print("Consider creating it with makeSolver.py")
        return
    myMPCPlanner.concretize()
    myMPCPlanner.reset()
    n = myMPCPlanner.n()
    staticGoalDict = {
        "m": 2, "w": 1.0, "prime": True, 'indices': [0, 1], 'parent_link': 0, 'child_link': n,
        'desired_position': [8, 0], 'epsilon': 0.2, 'type': "staticSubGoal", 
    }
    staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
    myMPCPlanner.setObstacles([sphereObst1], 0.5)
    myMPCPlanner.setGoal(staticGoal)
    if robotType in ['diffDrive', 'boxer']:
        env = gym.make(envName, render=True, dt=myMPCPlanner.dt())
    else:
        env = gym.make(envName, render=True, dt=myMPCPlanner.dt(), n=myMPCPlanner.n())
    limits = np.array([[-50, ] * n, [50, ] * n])
    myMPCPlanner.setJointLimits(limits)
    q0 = np.random.random(n)
    #q0 = np.array([-8.0, 0.0, 0.0])
    ob = env.reset(pos=q0)
    env.add_obstacle(sphereObst1)
    env.add_goal(staticGoal)
    n_steps = 1000
    for i in range(n_steps):
        q, qdot, vel = extract_joint_states(ob, robotType)
        if robotType in ['diffDrive', 'boxer']:
            action = myMPCPlanner.computeAction(q, qdot, vel)
        else:
            action = myMPCPlanner.computeAction(q, qdot)
        ob, *_ = env.step(action)

def extract_joint_states(ob: dict, robotType: str):
    if  robotType == 'boxer':
        q = ob['joint_state']['position']
        qdot = ob['joint_state']['velocity']
        vel = None
        if robotType in ['diffDrive', 'boxer']:
            vel = np.array([ob['joint_state']['forward_velocity'], qdot[2]])
    else:
        q = ob['x']
        qdot = ob['xdot']
        vel = None
        if robotType in ['diffDrive']:
            vel = ob['vel']
    return q, qdot, vel


if __name__ == "__main__":
    main()
