import time

from banana_env import BananaEnv
from cart_pole_env import CartPoleEnv

from q_agent import QAgent


#env = BananaEnv()
env = CartPoleEnv()
env.reset()

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())
#[50,20,10]
agent = QAgent(state_space= env.get_state_space_size(), action_space=env.get_action_space_size(), layers=[50,50, 50],
               mem_size=10000)
env.reset()
curr_score = 0
score_list = []
running_score = 0
start_eps=0.1
eps=start_eps
eps_decrease=0.95
for i in range(10000):
    if i%100 == 0:
        eps*=eps_decrease
    env.reset()
    done=False
    curr_score=0
    while not done:
        exp = agent.act(env, eps)
        curr_score = curr_score + exp.reward
        done =exp.done
        agent.learn(5)
    running_score = 0.95*running_score+0.05*curr_score
    if i%10:
        print(running_score)
        agent.update_target()


print("done")
