import numpy as np


def irt_2pl(theta, a, b):
    """
    Two-Parameter Logistic IRT probability.
    """
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))
