import numpy as np
import yaml
import os
import sys
sys.path.insert(0, '/Users/Alex/develop/forces_pro_client')
import forcespro
import gym
import re
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
import robotmpcs
from robotmpcs.planner.mpcPlanner import MPCPlanner
import urdfenvs.boxer_robot
import urdfenvs.panda_reacher

import time

path_name = os.path.dirname(os.path.realpath(__file__)) + '/'


def main():
    CONFIGPATH="config/pandaMPC.yaml"
    test_setup = os.path.dirname(os.path.realpath(__file__)) + "/" + CONFIGPATH
    robotType = re.findall('\/(\S*)M', CONFIGPATH)[0]
    solversDir = os.path.dirname(os.path.realpath(__file__)) + "/solvers/"
    try:
        myMPCPlanner = MPCPlanner(test_setup, robotType, solversDir)
    except SolverDoesNotExistError as e:
        print(e)
        print("Consider creating it with makeSolver.py")
        return

    n_epochs = 10
    n_steps = 1_000
    myMPCPlanner.reset()
    n = myMPCPlanner.n()

    staticGoalDict = {
        "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': n,
        'desired_position': [2, -3, 0], 'epsilon': 0.1, 'type': "staticSubGoal",
    }
    staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
    myMPCPlanner.setGoal(staticGoal)

    #env = gym.make("boxer-robot-vel-v0", dt=0.01, render=True)
    env = gym.make("panda-reacher-vel-v0", dt=0.01, render=True)
    limits = np.array([[-50, ] * n, [50, ] * n])  # for point robot this is only "x"
    myMPCPlanner.setJointLimits(limits)
    myMPCPlanner.concretize()
    q0 = np.array([0, 3])
    ob = env.reset(pos=q0)
    #durations = []
    for ep in range(n_epochs):
        ob = env.reset()
        staticGoalDict['desired_position'] = [np.random.uniform(0.0, 0.5),
                                              np.random.uniform(-0.5, 0.5),
                                              np.random.uniform(0.0, 0.5)]
        staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
        env.add_goal(staticGoal)
        myMPCPlanner.setGoal(staticGoal)

        for i in range(n_steps):
            q = ob['x']
            qdot = ob['xdot']
            #action = 0.1 * env.action_space.sample()
            action = myMPCPlanner.computeAction(q, qdot)

            ob, reward, done, info = env.step(action)
            if i % 10_000 == 1:
                print(ep, i, action, ob, reward)

            if done:
                print(done, i)

if __name__ == "__main__":
    main()
