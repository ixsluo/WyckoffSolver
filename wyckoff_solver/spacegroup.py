from importlib import resources

import numpy as np

from .utils.singleton import ParamEncodeBase, ParamSingletonFactory


with resources.open_text('wyckoff_solver.resources', 'spg_multiplicity.txt') as f:
    spg_multiplicity = [np.array(list(map(int, line.strip().split(',')))) for line in f.readlines()]
with resources.open_text('wyckoff_solver.resources', 'spg_variability.txt') as f:
    spg_variability = [np.array(list(map(int, line.strip().split(','))), dtype=bool) for line in f.readlines()]


class SpaceGroup(ParamEncodeBase, metaclass=ParamSingletonFactory.create_metaclass("SpaceGroup")):
    @classmethod
    def encode_arguments(cls, spgno, **kwargs):
        return spgno

    def __init__(self, spgno: int):
        if not (1 <= spgno <= 230):
            raise ValueError("spacegroup number must 1 to 230.")
        self.spgno = spgno
        self.multiplicity = spg_multiplicity[spgno - 1]
        self.variability = spg_variability[spgno - 1]
