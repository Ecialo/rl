from itertools import product

from common.dynamic import (
    strategy_iteration,
    Strategy,
)

A = list(range(-5, 6))
Sp = {(i, j) for i, j in product(range(21), range(21))}
S = Sp
gamma = 0.9


def P(s, a, ss):
    pass