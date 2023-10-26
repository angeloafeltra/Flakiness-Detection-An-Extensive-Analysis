from abc import ABC, abstractmethod

import numpy as np


class Validator(ABC):


    @abstractmethod
    def validation(self,pipeline): pass

    def filtro(self,y_set):
        tf=np.count_nonzero(y_set.to_numpy())
        tnf=y_set.to_numpy().shape[0]-np.count_nonzero(y_set.to_numpy())
        if tf>tnf or tf<20: return False
        return True

