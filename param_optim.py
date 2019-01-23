import random
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
elif env_sel == "CartPole":
    env = CartPoleEnv()
else:
    raise ValueError("specified environment " + env_sel + " does not match any available environment")
env.reset()

layers_choices = [[32, 32], [64, 64], [256, 128], [32, 32, 32], [128, 64]]
mem_size_choices = [200, 1000, 5000, 20000]
update_every_choices = [1, 2, 5, 10]
learn_every_choices = [1, 2, 4, 8, 16]
learning_rate_choices = [0.001, 0.0005, 0.0002]
eps_decay_choices = [0.99, 0.995, 0.997]
double_qnet_choices = [True, False]
delayer_choices = [True, False]

num_tests = 100

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())
# [50,20,10]
results = []
result_pars = []
for test_i in range(num_tests):
    pars = {}
    pars["layers_sel"] = random.choice(layers_choices)
    pars["mem_size_sel"] = random.choice(mem_size_choices)
    pars["update_every_sel"] = random.choice(update_every_choices)
    pars["learn_every_sel"] = random.choice(learn_every_choices)
    pars["learning_rate_sel"] = random.choice(learning_rate_choices)
    pars["eps_decay_sel"] = random.choice(eps_decay_choices)
    pars["double_qnet_sel"] = random.choice(double_qnet_choices)
    pars["delayer_sel"] = random.choice(delayer_choices)

    print(">>> test "+str(test_i))
    print(">>> parameters:")
    print(pars)

    agent = QAgent(state_space=env.get_state_space_size(),
                   action_space=env.get_action_space_size(),
                   layers=pars["layers_sel"],
                   mem_size=pars["mem_size_sel"],
                   use_delayer=pars["delayer_sel"],
                   learning_rate=pars["learning_rate_sel"],
                   double_qnet=pars["double_qnet_sel"])

    env.reset()

    update_every = pars["update_every_sel"]
    learn_every = pars["learn_every_sel"]
    curr_score = 0
    score_window = deque(maxlen=100)  # last 100 scores
    score_list = []
    mean_score_list = []
    running_score = 0
    eps_start = 1.0
    eps_decay = pars["eps_decay_sel"]
    eps_end = 0.01
    eps = eps_start
    max_ep_len = 400
    train_episodes = 701
    for episode in range(train_episodes):
        eps = max(eps * (eps_decay), eps_end)
        env.reset()
        done = False
        curr_score = 0
        for act_i in range(max_ep_len):
            exp = agent.act(env, eps)
            curr_score = curr_score + exp.reward
            if exp.done:
                break
            if act_i % learn_every == 0:
                agent.learn(64)
        score_list.append(curr_score)
        score_window.append(curr_score)
        if episode % update_every == 0:
            agent.update_target()
        if episode % 20 ==0:
            print("episode "+ str(episode) + ", mean score: " + str(np.mean(score_window)))
        if episode % 100 == 0:
            mean_score_list.append(np.mean(score_window))
    print("test completed with scores: "+str(mean_score_list))

    agent.save_checkpoint(local_checkpoint="test_out_3/qnet_" + env_sel + "_local_test_" + str(test_i) + ".ckp")
    pickle.dump((score_list, mean_score_list, pars), open("test_out_3/qnet_" + env_sel + "_scores_and_pars_test_" + str(test_i) + ".p", "wb"))


    results.append(max(mean_score_list))
    result_pars.append(pars)
