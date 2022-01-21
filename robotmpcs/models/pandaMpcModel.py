from robotmpcs.models.mpcModel import MpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator


class PandaMpcModel(MpcModel):

    def __init__(self, m, N):
        n = 7
        super().__init__(m, n, N)
        self._fk = FkCreator('panda', n).fk()
        self._modelName = 'panda'

