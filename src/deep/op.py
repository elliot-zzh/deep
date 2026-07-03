import numpy as np
from .tensor import *
from .graph import *

# TODO: should we implement add, sub, mul, div etc. here? (or just use np.add etc. directly)


def exp(input):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.exp, "__call__", input)


def log(input):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.log, "__call__", input)


# TODO: should we implement exp2, log2, log10 etc. ?


def sum(input: Tensor, axis=None, keepdims=False):
    return input.__array_ufunc__(np.add, "reduce", input, axis=axis, keepdims=keepdims)
