import numpy as np
import yaml
import os
import sys
import forcespro
import gym
import planarenvs.ground_robots
import planarenvs.point_robot
import planarenvs.n_link_reacher
import re
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
import robotmpcs
from robotmpcs.planner.mpcPlanner import MPCPlanner



path_name = os.path.dirname(os.path.realpath(__file__)) + '/'
envMap = {
    'planarArm': 'nLink-reacher-acc-v0', 
    'diffDrive': 'ground-robot-acc-v0', 
    'pointRobot': 'point-robot-acc-v0', 
}
obst1Dict = {
    "dim": 2,
    "type": "sphere",
    "geometry": {"position": [1.5, -0.2], "radius": 0.7},
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
    n = myMPCPlanner.dof()
    staticGoalDict = {
        "m": 2, "w": 1.0, "prime": True, 'indices': [0, 1], 'parent_link': 0, 'child_link': n,
        'desired_position': [2, -3], 'epsilon': 0.2, 'type': "staticSubGoal", 
    }
    staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
    myMPCPlanner.setObstacles([sphereObst1], 0.5)
    myMPCPlanner.setGoal(staticGoal)
    if robotType == 'diffDrive':
        env = gym.make(envName, render=True, dt=myMPCPlanner.dt())
    else:
        env = gym.make(envName, render=True, dt=myMPCPlanner.dt(), n=myMPCPlanner.dof())
    limits = np.array([[-50, ] * n, [50, ] * n])
    myMPCPlanner.setJointLimits(limits)
    q0 = np.random.random(n)
    ob = env.reset(pos=q0)
    env.addObstacle(sphereObst1)
    env.addGoal(staticGoal)
    n_steps = 1000
    for i in range(n_steps):
        q = ob['x']
        qdot = ob['xdot']
        if robotType == 'diffDrive':
            vel = ob['vel']
            action = myMPCPlanner.computeAction(q, qdot, vel)
        else:
            action = myMPCPlanner.computeAction(q, qdot)
        ob, *_ = env.step(action)


if __name__ == "__main__":
    main()
