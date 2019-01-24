class DiscreteActonEnv:
    """
     Abstract class for a wrapper to an environment with a discrete action space (and a continuous state space). The
     interface is meant to match closely the one defined for gym environment.
    """
    def __init__(self, seed):
        raise NotImplementedError("Constructor must be implemented")

    def get_env(self):
        raise NotImplementedError("get_env method must be implemented and return the wrapped environment")

    def get_state_space_size(self):
        raise NotImplementedError(
            "get_state_space_size method must be implemented and return the number of variables in state space")

    def get_action_space_size(self):
        raise NotImplementedError(
            "get_action_space_size method must be implemented and returnthe number of discrete actions")

    def get_state(self):
        raise NotImplementedError("get_state method must be implemented and retun the current environment state")

    def reset(self):
        raise NotImplementedError(
            "reset() method must be implemented, reset the underlying environment and retun the new state after reset")

    def step(self, action):
        raise NotImplementedError(
            "step(action) method must be implemented, execute action and return a tuple (new state, reward, done)")

    def close(self):
        raise NotImplementedError("close() method must be implemented")

    def seed(self, seed=None):
        raise NotImplementedError("seed(seed=None) method must be implemented")

    def render(self):
        raise NotImplementedError("render() method must be implemented")
