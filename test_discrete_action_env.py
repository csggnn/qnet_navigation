from cart_pole_env import CartPoleEnv
from banana_env import BananaEnv

for env in [CartPoleEnv(), BananaEnv()]:
    print("state")
    print(env.get_state())
    print("action_space_size")
    print(env.get_action_space_size())
    print("state_space_size")
    print(env.get_state_space_size())
    print("reset")
    print(env.reset())
    print("step")
    print(env.step(0))
    print("env")
    print(env.get_env())
    print("close")
    print(env.close())
