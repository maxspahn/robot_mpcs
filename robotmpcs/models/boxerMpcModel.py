from robotmpcs.models.diffDriveMpcModel import DiffDriveMpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator


class BoxerMpcModel(DiffDriveMpcModel):

    def __init__(self, time_horizon):
        dim_goal = 2
        super().__init__(dim_goal, time_horizon)
        self._fk = FkCreator('boxer', 3).fk()
        self._modelName = 'boxer'
