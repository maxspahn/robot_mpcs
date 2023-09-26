import os
import sys
import re
import yaml
from robotmpcs.models.diff_drive_mpc_model import MpcDiffDriveModel
from robotmpcs.models.mpcModel import MpcModel

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup

def main(robot_type, setup_file):
    setup = parse_setup(setup_file)
    if setup['robot']['base_type'] == 'holonomic':
        mpc_model = MpcModel(initParamMap=True, **setup)
    elif setup['robot']['base_type'] == 'diffdrive':
        mpc_model = MpcDiffDriveModel(initParamMap=True, **setup)
    mpc_model.setModel()
    mpc_model.setCodeoptions()
    path_to_solvers = os.path.dirname(os.path.abspath(__file__)) + '/solvers/'
    mpc_model.generateSolver(location=path_to_solvers)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a config file for solver generation.")
    else:
        robot_type = re.findall('\/(\S*)M', sys.argv[1])[0]
        main(robot_type, sys.argv[1])
