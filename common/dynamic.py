from collections import defaultdict
from itertools import product

import numpy


def estimate(pi, S, Sp, R, A, P, theta=0.01, gamma=0.9):
    V = {}
    for s in Sp:
        V[s] = 0.0
    while True:
        d = 0.0
        for s in S:
            v = V[s]
            sum_ = 0.0
            for a in A:
                for ss in Sp:
                    sum_ += pi[(s, a)] * P(s, a, ss) * (R[(s, a, ss)] + gamma * V[ss])
            V[s] = sum_
            d = max(d, abs(v - V[s]))
        if d < theta:
            break
    return V


def test():
    A = {'up', 'down', 'left', 'right'}
    Sp = {(i, j) for i, j in product(range(4), range(4))}
    S = Sp - {(0, 0), (3, 3)}
    R = defaultdict(lambda: -1.0)
    pi = defaultdict(lambda: 0.25)

    def P(s, a, ss):
        sx, sy = s
        tx, ty = ss
        if a == 'up' and sx == tx:
            if ty - sy == -1:
                return 1.0
            elif sy == 0 and ty == 0:
                return 1.0
            else:
                return 0.0
        elif a == 'down' and sx == tx:
            if ty - sy == 1:
                return 1.0
            elif sy == 3 and ty == 3:
                return 1.0
            else:
                return 0.0
        elif a == 'left' and sy == ty:
            if tx - sx == -1:
                return 1.0
            elif sx == 0 and tx == 0:
                return 1.0
            else:
                return 0.0
        elif a == 'right' and sy == ty:
            if tx - sx == 1:
                return 1.0
            elif sx == 3 and tx == 3:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    V = numpy.ndarray((4, 4))
    V_res = estimate(pi, S, Sp, R, A, P, theta=0.001, gamma=1.0)
    for s, v in V_res.items():
        V[s] = v
    print(V)

if __name__ == '__main__':
    test()