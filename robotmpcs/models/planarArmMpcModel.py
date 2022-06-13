from robotmpcs.models.mpcModel import MpcModel
from forwardkinematics.fksCommon.fk_creator import FkCreator


class PlanarArmMpcModel(MpcModel):

    def __init__(self, dim_goal, dof, time_horizon):
        super().__init__(dim_goal, dof, time_horizon)
        self._fk = FkCreator('planarArm', dof).fk()
        self._modelName = "planarArm"

