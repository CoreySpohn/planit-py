"""Set of realizations of an orbit.

This takes in a set of parameters with uncertainties attached and generates a
set of realizations of the orbit by sampling the parameters.
"""


class OrbitRealizationSet:
    """A set of realizations of an orbit."""

    def __init__(self, a, e, inc, W, w):
        self.a = a
        self.e = e
        self.inc = inc
        self.W = W
        self.w = w
