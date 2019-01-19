from experience_replayer import ExperienceReplayer
from pytorch_base_network import PyTorchBaseNetwork
import numpy as np
from collections import namedtuple
from torch import nn, optim
import torch


class QAgent:
    """ Agent using a neural network and experience replay to implement Q learning

    The agent should:
    - act: use its neural network select actions according to its Q policy and memorize experiences
    - learn: update its neural network parameters according to training experiences loaded from its memory

    """

    def __init__(self, state_space, action_space, layers=[100, 100], mem_size = 1000):
        self.mem = ExperienceReplayer(mem_size)
        self.gamma = 0.95
        self.state_space = state_space
        self.action_space = action_space
        # The local network is updated at every steps, but its evolution is not immediately used in selecting actions.
        # In Double DQN, the local network is also used to extract action values used for updates, although the actual
        # actions are selected depending on the target network.
        self.qnet_local = PyTorchBaseNetwork(input_shape = (state_space,),lin_layers = layers, output_shape=(action_space,))
        # The agent inspects the environment using the target network to decide on its actions. the target network also
        # selects the best action in each state for the pourpose of updating the local network.
        self.qnet_target = PyTorchBaseNetwork(input_shape = (state_space,),lin_layers = layers, output_shape=(action_space,))
        # As it is beneficial to have a target network which is different from (delayed w.r.t.) the local network, why
        # is it okay to have them identical or almost identical just after target network gets updated?
        # I am introducing a 3rd network, which acts as a buffer between local and target network.
        # every N learn calls, delayer weigths will be moved to the target network and then local weigths will be copied
        # over to delayer.
        # there will thus always be a distance of N to 2*N update calls between qnet_local and qnet_target
        self.qnet_delayer = PyTorchBaseNetwork(input_shape = (state_space,),lin_layers = layers, output_shape=(action_space,))

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=0.003)

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def act(self, env, eps):
        """ Act on provided environment.

        The agent selects an action according to epsilon greedy policy using its local network. It then performs the
        selected action on the environment and stores the old state, action, reward, new state and done flag into its
        experience buffer

        Params
        ======
            state (array): environent stae or observation before performing action
            eps (float): epsilon parameter for epsilon greedy policy. Totally random actions if eps=1, greedy if eps=0
            env (class object): environment in which agent operates. will be used by env_step_fun
            env_step_fun (function): calls env.step(env, action) and returns a (next_state, reward, done) tuple. As
                different environments can provide env.step() methods with different signature, this method requires an
                env_step_fun functions able to call env.step for the specific environment and to collect output in a
                standardized format.

        """
        state = env.get_state()
        action=self.select_action(state, eps)
        next_state, reward, done = env.step(action)
        self.mem.store(self.experience(state, action, reward, next_state, done))
        return self.experience(state, action, reward, next_state, done)

    def select_action(self, state, eps):
        """ Select an action according to local network """
        greedy =  np.random.rand()>eps
        if greedy:
            act = np.argmax(self.qnet_target.forward_np(state))
        else:
            act = np.random.choice(self.action_space)
        return act

    def learn(self, batch_size):
        """


        :return:
        """
        experiences = self.mem.draw(batch_size)

        if experiences is None:
            return None

        next_states = torch.tensor([exp.next_state for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences])
        states = torch.tensor([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences])

        # Double DQN : find max in a network, pick value from the other

        best_actions = np.argmax(self.qnet_local.forward(next_states.float()).detach(), 1).unsqueeze(1)
        best_actvalues = self.qnet_target.forward(next_states.float()).detach().gather(1, best_actions)

        target = rewards + self.gamma * (best_actvalues) *(1.0-dones.float())
        current = self.qnet_local.forward(states.float()).gather(1, actions.unsqueeze(1))

        loss = nn.MSELoss()
        loss_curr = loss(current, target)
        self.optimizer.zero_grad()
        loss_curr.backward()
        self.optimizer.step()

        return loss_curr


    def update_target(self):

        self.qnet_target.load_state_dict(self.qnet_local.state_dict())
        #self.qnet_target.load_state_dict(self.qnet_delayer.state_dict())

    def save_checkpoint(self,  target_checkpoint, local_checkpoint = None, delayer_checkpoint=None):
        self.qnet_target.save_model(target_checkpoint, "q_agent_target_net")
        if local_checkpoint is not None:
            self.qnet_local.save_model(local_checkpoint, "q_agent_local_net")
        if delayer_checkpoint is not None:
            self.qnet_delayer.save_model(delayer_checkpoint, "q_agent_delayer_net")

    def load_checkpoint(self, target_checkpoint, local_checkpoint = None, delayer_checkpoint=None):
        if local_checkpoint is None:
            local_checkpoint=target_checkpoint
        if delayer_checkpoint is None:
            delayer_checkpoint=local_checkpoint
        self.qnet_target = PyTorchBaseNetwork(target_checkpoint)
        self.qnet_local = PyTorchBaseNetwork(local_checkpoint)
        self.qnet_delayer = PyTorchBaseNetwork(delayer_checkpoint)





