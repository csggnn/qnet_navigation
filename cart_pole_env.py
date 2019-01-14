import gym
from discrete_action_env import DiscreteActonEnv


class CartPoleEnv(DiscreteActonEnv):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state = self.env.reset()

    def get_env(self):
        return self.env

    def get_state_space_size(self):
        return self.env.observation_space.shape[0]

    def get_action_space_size(self):
        return self.env.action_space.n

    def get_state(self):
        return self.state

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        self.state , reward, done,_ = self.env.step(action)
        return (self.state, reward, done)

    def close(self):
        self.env.close()