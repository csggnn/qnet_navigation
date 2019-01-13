import time

from unityagents import UnityEnvironment

from q_agent import QAgent, banana_env_step

env = UnityEnvironment(
    file_name="/media/csggnn/OS/Users/giann/Projects/courses/reinf_learn_udacity/deep-r-learn/p1_navigation/Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = QAgent(state_space=brain.vector_observation_space_size, action_space=brain.vector_action_space_size,
               mem_size=10000)

# state_t = torch.tensor(state).float()

for i in range(1000):
    exp = agent.act(state, 0.1, env, banana_env_step)
    time.sleep(0.001)
    if exp.done:
        env.reset()

env.reset()
curr_score = 0
score_list = []
running_score = 0
for i in range(10000):
    for j in range(50):
        exp = agent.act(state, 0.1, env, banana_env_step)
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
    exp = agent.act(state, 0.1, env, banana_env_step)
    time.sleep(0.001)
    if exp.done:

        env.reset()

print(exp)

print("done")
