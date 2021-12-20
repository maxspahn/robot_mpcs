from robot_mpcs.mpcModel import MpcModel
from forwardKinematics.fksCommon.fk_creator import FkCreator


class PlanarArmMpcModel(MpcModel):

    def __init__(self, m, N, n):
        super().__init__(m, n, N)
        self._fk = FkCreator('planarArm', n).fk()

