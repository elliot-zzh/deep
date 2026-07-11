from abc import ABC, abstractmethod

import numpy as np

from .op import *
from .tensor import *


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass
