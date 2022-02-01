from robotmpcs.models.diffDriveMpcModel import DiffDriveMpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator


class BoxerMpcModel(DiffDriveMpcModel):

    def __init__(self, N):
        super().__init__(2, N)
        self._fk = FkCreator('boxer', 3).fk()
        self._modelName = 'boxer'

