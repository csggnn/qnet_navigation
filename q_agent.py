from experience_replayer import ExperienceReplayer
from pytorch_base_network import PyTorchBaseNetwork
import numpy as np

class QNetwork:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.nnet = PyTorchBaseNetwork(input_shape = (state_space,),lin_layers = [100,100], output_shape=(action_space,))

    def select_action(self, state, eps=0.1):
        """ select action according to eps greedy policy
        """
        greedy =  np.random.rand()>eps
        if greedy:
            _, act = max(self.nnet.forward(state))
        else:
            act =np.random.choice(self.action_space)
        return act

    def update(self):
        None

class QAgent:
    """ Agent using a neural network and experience replay to implement Q learning

    The agent should:
    - act: use its neural network select actions according to its Q policy and memorize experiences
    - learn: update its neural network parameters according to training experiences loaded from its memory

    """

    def __init__(self, state_space, action_space,  mem_size = 1000):
        self.mem = ExperienceReplayer(mem_size)
        self.qnet = QNetwork(input_shape = (state_space,),lin_layers = [100,100], output_shape=(action_space,))






