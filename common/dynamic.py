import numpy
from itertools import product


def estimate(pi, S, Sp, R, A, theta=0.01, gamma=0.9):
    V = {}
    for s in Sp:
        V[s] = 0.0
    while True:
        d = 0.0
        for s in S:
            v = V[s]
            sum_ = 0.0
            for a in A:
                for ss, P in pi[(s, a)]:
                    sum_ += P*(R[(s, a, ss)] + gamma*V[ss])
            V[s] = sum_
            d = max(d, abs(v - V[s]))
        if d < theta:
            break
    return V


def test():
    A = {'up', 'down', 'left', 'right'}
    Sp = {(i, j) for i, j in product(range(4), range(4))}
    S = Sp - {(0, 0), (3, 3)}
