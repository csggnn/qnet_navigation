import gym
from discrete_action_env import DiscreteActonEnv

class DiscreteGymEnv(DiscreteActonEnv):
    """
     Light wrapper to an discrete gym environment.
    """
    def __init__(self, envName, seed=None):
        self.env = gym.make(envName)
        self.env.seed(seed)
        self.state=self.env.reset()

    def get_env(self):
        return self.env

    def get_state_space_size(self):
        return self.env.observation_space.shape[0]

    def get_action_space_size(self):
        return self.env.action_space.n

    def get_state(self):
        return self.state

    def reset(self):
        self.state=self.env.reset()
        return self.state

    def step(self, action):
        out = self.env.step(action)
        self.state = out[0]
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def render(self):
        return self.env.render()
