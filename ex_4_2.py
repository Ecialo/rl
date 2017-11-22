from itertools import product, accumulate
from operator import mul
from math import *

from common.dynamic import (
    strategy_iteration,
    Strategy,
)

factor = list(accumulate(range(1, 21), mul))


def poisson(n, l):
    return ((l**n)/factor[n - 1])*e**(-l)


A = list(range(-5, 6))
Sp = {(i, j) for i, j in product(range(21), range(21))}
S = Sp
gamma = 0.9


def P(s, a, ss):
    pass


class R:

    def __getitem__(self, item):
        s, a, ss = item