from collections import defaultdict
from itertools import product
from random import choice

import numpy


def argmax(f, args):
    return max(args, key=f)


def estimate(pi, S, Sp, R, A, P, V, theta=0.01, gamma=1.0):
    i = 0
    while True:
        d = 0.0
        for s in S:
            v = V[s]
            sum_ = 0.0
            for a in A:
                for ss in Sp:
                    sum_ += pi[(s, a)] * P(s, a, ss) * (R[(s, a, ss)] + gamma * V[ss])
            V[s] = sum_
            # print(s, v, V[s])
            d = max(d, abs(v - V[s]))
        # print(i, d)
        i += 1
        if d < theta:
            break
    return V


def improve(pi, S, A, P, R, V, gamma=1.0):

    def mA(s):

        def pS(a):
            sum_ = 0
            for ss in S:
                sum_ += P(s, a, ss)*(R[(s, a, ss)] + gamma*V[ss])
            return sum_

        return pS

    is_stable = True
    for s in S:
        b = pi[s]
        amax = argmax(mA(s), A)
        # print("AMAX", amax)
        pi[s] = amax
        if b != pi[s]:
            is_stable = False
    return is_stable, pi


class Strategy:

    def __init__(self, A, ds=None):
        self._ds = ds if ds is not None else defaultdict(lambda: 1/len(A))
        self._A = A
        self._s = {}

    def __getitem__(self, item):
        # print(item, isinstance(item[-1], tuple))
        if isinstance(item[0], tuple):
            # print(self._ds[item])
            return self._ds[item]
        else:
            if item in self._s:
                return self._s[item]
            else:
                return choice(self._A)

    def __setitem__(self, key, value):
        # print(key, value)
        if isinstance(key[0], tuple):
            self._ds[key] = value
        else:
            # print(key, value)
            self._s[key] = value
            for a in self._A:
                self._ds[(key, a)] = 1.0 if a == value else 0.0
                # print(self._ds)

    def items(self):
        return self._s.items()


def strategy_iteration(R, A, S, Sp, P):
    V = {}
    pi = Strategy(A)
    i = 0
    for s in Sp:
        V[s] = 0.0
        # det strategy ruins everything
        # pi[s] = choice(A)
    while True:
        V = estimate(pi, S, Sp, R, A, P, V)
        is_stable, pi = improve(pi, S, A, P, R, V)
        print(i)
        if is_stable:
            # print(pi._s)
            return pi
        i += 1


def test():
    A = ['u', 'd', 'l', 'r']
    Sp = {(i, j) for i, j in product(range(4), range(4))}
    S = Sp - {(0, 0), (3, 3)}
    R = defaultdict(lambda: -1.0)
    pi = Strategy(A)

    # print(argmax(lambda x: x**2, [3,2,1]))

    def P(s, a, ss):
        sx, sy = s
        tx, ty = ss
        if a == 'u' and sx == tx:
            if ty - sy == -1:
                return 1.0
            elif sy == 0 and ty == 0:
                return 1.0
            else:
                return 0.0
        elif a == 'd' and sx == tx:
            if ty - sy == 1:
                return 1.0
            elif sy == 3 and ty == 3:
                return 1.0
            else:
                return 0.0
        elif a == 'l' and sy == ty:
            if tx - sx == -1:
                return 1.0
            elif sx == 0 and tx == 0:
                return 1.0
            else:
                return 0.0
        elif a == 'r' and sy == ty:
            if tx - sx == 1:
                return 1.0
            elif sx == 3 and tx == 3:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    V = {}
    for s in Sp:
        V[s] = 0.0
    V_res = estimate(pi, S, Sp, R, A, P, V, theta=0.001, gamma=1.0)

    V = numpy.ndarray((4, 4))
    for s, v in V_res.items():
        V[s] = v
    print(V)

    print("\n\n")

    result_strategy = strategy_iteration(R, A, S, Sp, P)
    result_table = [[None for _ in range(4)] for _ in range(4)]
    for s, a in result_strategy.items():
        print(s, a)
        y, x = s
        result_table[x][y] = a

    for row in result_table:
        print(row)

if __name__ == '__main__':
    test()