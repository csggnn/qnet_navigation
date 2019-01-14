import time

from banana_env import BananaEnv
from cart_pole_env import CartPoleEnv

from q_agent import QAgent


env = CartPoleEnv()
env.reset()

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())

agent = QAgent(state_space= env.get_state_space_size(), action_space=env.get_action_space_size(), layers=[20],
               mem_size=10000)
env.reset()
curr_score = 0
score_list = []
running_score = 0
for i in range(10000):
    for j in range(50):
        exp = agent.act(env, 0.1)
        curr_score = curr_score + exp.reward
        if exp.done:
            running_score = 0.99 * running_score + 0.01*curr_score
            score_list.append(curr_score)
            curr_score = 0
            env.reset()
        agent.learn(20)
    agent.update_target()
    print(running_score)

for i in range(1000):
    exp = agent.act(env, 0.1)
    time.sleep(0.001)
    if exp.done:

        env.reset()

print(exp)

print("done")
