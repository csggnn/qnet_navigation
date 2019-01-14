from cart_pole_env import CartPoleEnv

env = CartPoleEnv()
print(env.get_state())
print(env.get_action_space_size())
print(env.get_state_space_size())
print(env.reset())
print(env.step(0))
print(env.get_env().observation_space)
print(env.close())
