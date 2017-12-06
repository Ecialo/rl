from collections import defaultdict
from itertools import product, accumulate
from functools import lru_cache
from operator import mul
from math import *

from scipy.stats import poisson as poi

poisson = poi.pmf

from common.dynamic import (
    strategy_iteration,
    Strategy,
)
from common.agent import Environment
from common.dynamic import DPAgent

# factor = list(accumulate(range(1, 22), mul))
#
#
# @lru_cache(maxsize=2048)
# def poisson(n, l):
#     return ((l**n)/factor[n - 1])*exp(-l)

MOVE_COST = 2
CAR_COST = 10
MAX_CARS = 10
LN1, LN2, LR1, LR2 = 3, 4, 3, 2

A = list(range(-5, 6))
Sp = {(i, j) for i, j in product(range(MAX_CARS+1), range(MAX_CARS+1))}
S = Sp
gamma_ = 0.9


class Rental(Environment):

    def __init__(
            self,
            mu_need_1, mu_need_2,
            mu_ret_1, mu_ret_2,
            cost_per_trans, income_per_car,
            max_car, max_trans
    ):
        self.mu_need_1 = mu_need_1
        self.mu_need_2 = mu_need_2
        self.mu_ret_1 = mu_ret_1
        self.mu_ret_2 = mu_ret_2

        self.cost_per_car_trans = cost_per_trans
        self.income_per_car = income_per_car

        self.max_car = max_car
        self.max_trans = max_trans

        self.states = list(
            product(
                range(max_car + 1),
                range(max_car + 1),
            )
        )
        self.num_states = (max_car + 1) ** 2

        self.actions = list(range(-max_trans, max_trans+1))
        self.num_actions = max_trans*2 + 1

        self.pre = self.precalc()

    def allowed_actions(self, state_id):
        s0, s1 = self.states[state_id]
        t1to2 = min(self.max_trans, s0)
        t2to1 = max(-self.max_trans, -s1)
        actions = list(range(t2to1, t1to2 + 1))
        aids = [i + self.max_trans for i in actions]
        # print(aids)
        return aids

    def reward(self, state_id, action_id, next_state_id):
        return self.pre['r'][(state_id, action_id)][next_state_id]

    def state_distribution(self, state_id, action_id):
        return list(self.pre['p'][(state_id, action_id)].items())

    def precalc(self):
        r1 = self.rewards_and_probs(self.mu_need_1, self.mu_ret_1)
        r2 = self.rewards_and_probs(self.mu_need_2, self.mu_ret_2)
        result = {
            'r': {},
            'p': {},
        }
        for sid in range(self.num_states):
            state = self.states[sid]
            s0, s1 = state
            for aid in self.allowed_actions(sid):
                action = self.actions[aid]
                r = result['r'][(sid, aid)] = defaultdict(float)
                p = result['p'][(sid, aid)] = defaultdict(float)
                cost = self.cost_per_car_trans * abs(action)
                n_beg_1 = s0 - action
                n_beg_2 = s1 + action

                for n_end1 in range(self.max_car + 1):
                    for n_end2 in range(self.max_car + 1):
                        sp1, sp2 = n_end1, n_end2
                        next_state = (sp1, sp2)
                        spid = self.states.index(next_state)
                        p1 = r1['p'][(n_beg_1, n_beg_2)]
                        p2 = r2['p'][(n_beg_2, n_beg_2)]
                        re1 = r2['r'][(n_beg_2, n_beg_2)]
                        re2 = r2['r'][(n_beg_2, n_beg_2)]
                        r[spid] += p1*p2*(cost + re1 + re2)
                        p[spid] += p1*p2
        return result

    def rewards_and_probs(self, mu_need, mu_ret):
        result = {
            'r': defaultdict(float),
            'p': defaultdict(float),
        }

        # max_car = self.max_car + self.max_trans
        max_car = self.max_car
        for n_beg in range(max_car + 1):
        # for n_beg in range(10*mu_ret + 1):
        #     for n_ret in range(10*mu_need + 1):
            for n_ret in range(max_car+1):
                for n_need in range(max_car+1):
                    prob = poisson(n_ret, mu_ret)*poisson(n_need, mu_need)
                    real_out = min(n_beg, n_need)

                    n_end = min(self.max_car, n_beg + n_ret - real_out)
                    key = (n_beg, n_end)
                    result['p'][key] += prob
                    result['r'][key] += prob * self.income_per_car * real_out
        return result



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
#
# pi = Strategy(A)
#
# result_strategy = strategy_iteration(R(), A, S, Sp, P, gamma=gamma_)
# result_table = [[None for _ in range(MAX_CARS + 1)] for _ in range(MAX_CARS + 1)]
# for s, a in result_strategy.items():
#     print(s, a)
#     y, x = s
#     result_table[x][y] = a
#
# for row in result_table:
#     print(row)

if __name__ == '__main__':
    env = Rental(
        3, 4,
        3, 2,
        -2, 10,
        7, 5
    )
    a = DPAgent(env)
    a.policy_iteration(20)
    print(a.V)
