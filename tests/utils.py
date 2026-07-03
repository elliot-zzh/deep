import numpy as np


def close(baseline, eval, eps=1e-12):
    return (baseline - eps < eval < baseline + eps).all()


def numerical_grad(func, input_, step=1e-6, use_log=True):
    if use_log:
        return np.exp(
            np.log(func(input_ + step) - func(input_ - step)) - np.log(step * 2)
        )  # central difference
    else:
        return (func(input_ + step) - func(input_ - step)) / (step * 2)
