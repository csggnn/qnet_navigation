from unityagents import UnityEnvironment
from discrete_action_env import DiscreteActonEnv

class BananaEnv(DiscreteActonEnv):
    """
    Interface to the banana environment.
    defines basic method so that i can substitute the banana environment with any other environment with continuous
    state space and discrete actions (and may relax this constraint in the future)
    """
    def __init__(self):
        self.env = UnityEnvironment(
    file_name="/media/csggnn/OS/Users/giann/Projects/courses/reinf_learn_udacity/deep-r-learn/p1_navigation/Banana_Linux/Banana.x86_64",
        no_graphics=True)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]

    def get_env(self):
        return self.env

    def get_state_space_size(self):
        return len(self.env_info.vector_observations[0])

    def get_action_space_size(self):
        return self.brain.vector_action_space_size

    def get_state(self):
        return self.env_info.vector_observations[0]

    def reset(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self.get_state()

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        reward = self.env_info.rewards[0]
        next_state = self.env_info.vector_observations[0]
        done = self.env_info.local_done[0]
        return (next_state, reward, done)

    def close(self):
        self.env.close()
