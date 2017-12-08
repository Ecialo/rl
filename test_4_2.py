import ex_4_2
max_car = 1

env = ex_4_2.Rental(
    1, 1,
    1, 1,
    1, 10,
    1, 1,
)


assert env.allowed_actions((0, 0)) == [0]
assert env.allowed_actions((0, 1)) == [-1, 0]
assert env.allowed_actions((1, 0)) == [0, 1]
assert env.allowed_actions((1, 1)) == [-1, 0, 1]


# # poisson distribution with a mu of 0 is a little too extreme, for test only
r = env.rewards_and_probs(0, 0)
assert r['p'][(0, 0)] == 1
assert r['r'][(0, 0)] == 0
#
r = env.rewards_and_probs(1, 1)
#
for k in env.pre['p']:
    # assert prob sum should all be 1
    print(env.pre['p'][k])
    assert abs(sum(env.pre['p'][k].values()) - 1) < 1e-6
#
# # Comparing rewards of (1, 0) & (2, 2), they should be symmetrical
# assert env.db['rewards'][(1, 0)][2] == env.db['rewards'][(2, 2)][1]
# assert env.db['probs'][(1, 0)][2] == env.db['probs'][(2, 2)][1]
#
#
# agent= DPAgent(env)
#
# # before training
# assert (agent.V == 0).all()
# assert agent.pi.tolist() == [1, 0, 1, 0]
#
# # after training via policy_iteration
# for i in range(5): agent.policy_iteration(num_iter=5)
# assert (agent.V == 0).all() == False
# assert (agent.pi == 1).all()
#
# # after reset
# agent.reset()
# assert (agent.V == 0).all()
# assert agent.pi.tolist() == [1, 0, 1, 0]
#
# # after training via value_iteration
# agent.value_iteration()
# assert (agent.V == 0).all() == False
# assert (agent.pi == 1).all()