import time

from banana_env import BananaEnv
from cart_pole_env import CartPoleEnv

from q_agent import QAgent


env_sel = "CartPole"

if env_sel=="Banana":
    env = BananaEnv()
elif env_sel =="CartPole":
    env = CartPoleEnv()
else:
    raise ValueError("specified environment "+ env_sel + " does not match any available environment")
env.reset()

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())
#[50,20,10]
agent = QAgent(state_space= env.get_state_space_size(), action_space=env.get_action_space_size(), layers=[256,64],
               mem_size=500)
env.reset()
curr_score = 0
score_list = []
running_score = 0
start_eps=0.3
eps=start_eps
eps_decrease=0.97
test_episodes = 3000
for episode in range(test_episodes):
    if episode%10 == 0:
        eps*=eps_decrease
    env.reset()
    done=False
    curr_score=0
    act_i=0
    while not done:
        act_i=act_i+1
        exp = agent.act(env, eps)
        curr_score = curr_score + exp.reward
        done =exp.done
        if act_i%4 ==0:
            agent.learn(64)
    running_score = 0.95*running_score+0.05*curr_score
    if episode%50==0:
        print("Episode "+ str(episode) + ". Eps = "+str(eps) +",  last score: " + str(curr_score) + ", running_score: " + str(running_score))
    if episode%10==0:
        agent.update_target()
    if episode%500 == 0:
        agent.save_checkpoint(target_checkpoint="qnet_"+env_sel+"_target_episode_"+str(episode)+".ckp")

agent.save_checkpoint(target_checkpoint="qnet_"+env_sel+"_target_final.ckp")

print("done")
