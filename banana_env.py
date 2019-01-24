from unityagents import UnityEnvironment
from discrete_action_env import DiscreteActonEnv

class BananaEnv(DiscreteActonEnv):
    """
    Gym-like interface to the banana environment.

    defines basic method so that BananaCollection environment can be tested just as any other gym environment with
    continuous state space and discrete actions (may relax this constraint in the future)

    Missing features:
        - close() method on UnityEnvironment has a bug. Ignoring close commands.
        - render() method is ignored. BananaEnv always renders in the current implementation.
        - environment defaults to train_mode=True in constructor. will need to add support for testing/exploitation.
    """
    def __init__(self, seed=None):
        self.env = UnityEnvironment(
    file_name="/media/csggnn/OS/Users/giann/Projects/courses/reinf_learn_udacity/deep-r-learn/p1_navigation/Banana_Linux/Banana.x86_64", seed=seed)
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
        if self.env._loaded is False:
            self.env = UnityEnvironment(
                file_name="/media/csggnn/OS/Users/giann/Projects/courses/reinf_learn_udacity/deep-r-learn/p1_navigation/Banana_Linux/Banana.x86_64")
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self.get_state()

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        reward = self.env_info.rewards[0]
        next_state = self.env_info.vector_observations[0]
        done = self.env_info.local_done[0]
        return (next_state, reward, done, self.env_info)

    def close(self):
        # workaround bug in Unity Environment: Calling env.close() once prevents instantiation of new environments #1167
        self.env.reset()

    def seed(self, seed=None):
        return seed

    def render(self):
        return


