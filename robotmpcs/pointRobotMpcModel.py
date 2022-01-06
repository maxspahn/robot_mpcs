from robotmpcs.mpcModel import MpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator


class PointRobotMpcModel(MpcModel):

    def __init__(self, m, N):
        n = 2
        super().__init__(m, n, N)
        self._fk = FkCreator('pointMass', n).fk()

