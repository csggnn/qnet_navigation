"""
Train a DQnet Agent on a gym or BananaCollection environment
"""

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from banana_env import BananaEnv # comment if unityagents is not available
from discrete_gym_env import DiscreteGymEnv # comment if gym is not available
from q_agent import QAgent
import pickle
from collections import namedtuple
import random




#select the environment on whch the agent is to be trained
env_sel = "banana"
#environment specific parameters
env_pars_tuple=namedtuple("env_pars", "name target_score n_episodes max_t")
env_par_dict = {"lunar": env_pars_tuple(name="Lunar Lander", target_score=200, n_episodes=2000, max_t=1000),
                  "cart": env_pars_tuple(name="Cart Pole", target_score=190, n_episodes=2000, max_t=300),
                  "banana": env_pars_tuple(name="Banana Collector", target_score=13.5, n_episodes=800, max_t=500)}

if env_sel is "lunar":
    env = DiscreteGymEnv('LunarLander-v2', 0)
elif env_sel is "banana":
    env = BananaEnv(0)
elif env_sel is "cart":
    env = DiscreteGymEnv('CartPole-v0', 0)
else:
    raise AttributeError("unknown environment selection " + env_sel)

env_pars=env_par_dict[env_sel]


state_size = env.get_state_space_size()
action_size = env.get_action_space_size()


#set dqnet and training parameters
layers = [128, 64] #hidden layers of the neural networks
mem_size = 5000 # capacity of the experience replay buffer, number of experiences
update_every = 2 # update target network every # episodes
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.99 # for epsilon greedy policy in training
learn_every = 4 # trigger learning every # actions
learning_rate=0.0005
use_delayer=True
double_qnet=True

agent = QAgent(state_space=env.get_state_space_size(),
               action_space=env.get_action_space_size(),
               layers=layers,
               mem_size=mem_size,
               learning_rate=learning_rate,
               use_delayer=use_delayer,
               double_qnet=double_qnet,
               seed=0)

#initialize
random.seed(0)
print(env.reset())
curr_score = 0
score_window = deque(maxlen=100)  # last 100 scores
score_list = []
running_score = 0
eps = eps_start
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

for episode in range(env_pars.n_episodes):
    eps = max(eps * eps_decay, eps_end) # reduce epsilon

    env.reset()
    done = False
    curr_score = 0
    act_i = 0

    for i in range(env_pars.max_t):
        act_i = act_i + 1
        exp = agent.act(env, eps)
        if episode % 20 == 0:
            env.render() # show, for gym environment only
        curr_score = curr_score + exp.reward
        if exp.done:
            break
        # learn every #learn_every actions
        if act_i % learn_every == 0:
            agent.learn(64)
    # update target network every #update_every episodes
    if episode % update_every == 0:
        agent.update_target()

    #save score
    score_list.append(curr_score)
    score_window.append(curr_score)

    #print and save
    if episode % 20 == 0:
        print("Episode " + str(episode) + ". Eps = " + str(eps) + ",  mean_score: " + str(np.mean(score_window)))
        ax.clear()
        ax.plot(np.arange(len(score_list)), score_list)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.draw()
        plt.pause(.001)
        if np.mean(score_window)>env_pars.target_score:
            agent.save_checkpoint(target_checkpoint="qnet_" + env_sel + "_target_episode_" + str(episode) + ".ckp",
                                  local_checkpoint="qnet_" + env_sel + "_local_episode_" + str(episode) + ".ckp",
                                  delayer_checkpoint="qnet_" + env_sel + "_delayer_episode_" + str(episode) + ".ckp")
            pickle.dump(score_list, open( "qnet_" + env_sel + "_scores_" + str(episode) + ".p", "wb"))

#final print and save
agent.save_checkpoint(target_checkpoint="qnet_" + env_sel + "_target_final.ckp")
pickle.dump(score_list, open( "qnet_" + env_sel + "_scores_final.p", "wb"))
plt.plot(np.arange(len(score_list)), score_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.draw()
plt.pause(.001)
print("done")
