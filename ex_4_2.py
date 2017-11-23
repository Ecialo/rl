from itertools import product, accumulate
from functools import lru_cache
from operator import mul
from math import *

from common.dynamic import (
    strategy_iteration,
    Strategy,
)

factor = list(accumulate(range(1, 21), mul))


@lru_cache(maxsize=2048)
def poisson(n, l):
    return ((l**n)/factor[n - 1])*exp(-l)

MOVE_COST = 2
CAR_COST = 10
MAX_CARS = 10
LN1, LN2, LR1, LR2 = 3, 4, 3, 2

A = list(range(-5, 6))
Sp = {(i, j) for i, j in product(range(MAX_CARS+1), range(MAX_CARS+1))}
S = Sp
gamma_ = 0.9


def P(s, a, ss):
    # print(s, a ,ss)
    # p = 1.0
    p = 0.0
    for need_1, need_2, ret_1, ret_2 in product(
            range(MAX_CARS + 1),
            range(MAX_CARS + 1),
            range(MAX_CARS + 1),
            range(MAX_CARS + 1),
    ):
        real_s = (
            s[0] - a + ret_1 - need_1,
            s[1] + a + ret_2 - need_2,
        )
        if real_s == ss:
            # print(poisson(need_1, LN1),
            #     poisson(need_2, LN2),
            #     poisson(ret_1, LR1),
            #     poisson(ret_2, LN2))
            p += (
                poisson(need_1, LN1) *
                poisson(need_2, LN2) *
                poisson(ret_1, LR1) *
                poisson(ret_2, LN2)
            )
            # print(p)
    # print(p)
    # return 1 - p
    return p


class R:

    def __getitem__(self, item):
        s, a, ss = item
        r = 0
        r -= abs(a)*MOVE_COST
        for need_1, need_2, ret_1, ret_2 in product(
                range(MAX_CARS),
                range(MAX_CARS),
                range(MAX_CARS),
                range(MAX_CARS),
        ):
            real_s = (
                s[0] - a + ret_1 - need_1,
                s[1] + a + ret_2 - need_2,
            )
            if real_s == ss:
                p = (
                    poisson(need_1, LN1) *
                    poisson(need_2, LN2) *
                    poisson(ret_1, LR1) *
                    poisson(ret_2, LN2)
                )
                r += p*(need_1 + need_2)*CAR_COST
        return r

pi = Strategy(A)

result_strategy = strategy_iteration(R(), A, S, Sp, P, gamma=gamma_)
result_table = [[None for _ in range(MAX_CARS + 1)] for _ in range(MAX_CARS + 1)]
for s, a in result_strategy.items():
    print(s, a)
    y, x = s
    result_table[x][y] = a

for row in result_table:
    print(row)