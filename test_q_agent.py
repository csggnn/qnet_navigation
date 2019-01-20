from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from banana_env import BananaEnv
from cart_pole_env import CartPoleEnv
from q_agent import QAgent

import pickle

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

env_sel = "Banana"

if env_sel == "Banana":
    env = BananaEnv()
    pars = {"layers": [128, 64], "mem_size": 5000, "train_episodes": 1200, "max_ep_len":1000, "update_every":2}
elif env_sel == "CartPole":
    env = CartPoleEnv()
    pars = {"layers": [128, 64], "mem_size": 2000, "train_episodes": 10000, "max_ep_len":500,"update_every":2}
else:
    raise ValueError("specified environment " + env_sel + " does not match any available environment")
env.reset()

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())
# [50,20,10]
agent = QAgent(state_space=env.get_state_space_size(), action_space=env.get_action_space_size(), layers=pars["layers"],
               mem_size=pars["mem_size"], use_delayer=True)
env.reset()
curr_score = 0
score_window = deque(maxlen=100)  # last 100 scores
score_list = []
running_score = 0
eps_start = 1.0
eps_decay = 0.995
eps_end = 0.01
eps = eps_start
max_ep_len = pars["max_ep_len"]
train_episodes = pars["train_episodes"]
update_every = pars["update_every"]
for episode in range(train_episodes):
    eps = max(eps * eps_decay, eps_end)
    env.reset()
    done = False
    curr_score = 0
    act_i = 0
    for i in range(max_ep_len):
        act_i = act_i + 1
        exp = agent.act(env, eps)
        curr_score = curr_score + exp.reward
        if exp.done:
            break
        if act_i % 4 == 0:
            agent.learn(64)
    score_list.append(curr_score)
    score_window.append(curr_score)
    if episode % 20 == 0:
        print("Episode " + str(episode) + ". Eps = " + str(eps) + ",  mean_score: " + str(np.mean(score_window)))
        ax.clear()
        ax.plot(np.arange(len(score_list)), score_list)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.draw()
        plt.pause(.001)
        if np.mean(score_window)>13:
            agent.save_checkpoint(target_checkpoint="qnet_" + env_sel + "_target_episode_" + str(episode) + ".ckp",
                                  local_checkpoint="qnet_" + env_sel + "_local_episode_" + str(episode) + ".ckp",
                                  delayer_checkpoint="qnet_" + env_sel + "_delayer_episode_" + str(episode) + ".ckp")
            pickle.dump(score_list, open( "qnet_" + env_sel + "_scores_" + str(episode) + ".p", "wb"))

    if episode % update_every == 0:
        agent.update_target()
    if episode % 2000 == 0:
        agent.save_checkpoint(local_checkpoint="qnet_" + env_sel + "_local_episode_" + str(episode) + ".ckp")



agent.save_checkpoint(target_checkpoint="qnet_" + env_sel + "_target_final.ckp")
plt.plot(np.arange(len(score_list)), score_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.draw()
plt.pause(.001)

print("done")
