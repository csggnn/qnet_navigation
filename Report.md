## Udacity Deep Reinforcement Learning Nanodegree
# Navigation Project Report

###Learning Algorithm

A Deep Q-Network has been implemented to solve the project according to [the paper on deep q network]. The following is 
a short overview of Deep Q-Networks and its building blocks. For a more detailed description please refer to the 
orignial papers []

####Short overview of Deep Q-Networks

In reinforcement learning, an **Agent** interacts with an **environment** by taking **actions**. Every time agent takes 
a new action, its **state** (or state observation) is updated and a (possibly 0) **reward** is collected.
 
Actions are selected by an agent depending on its state according to a policy. 
**Q-Learning** is a reinforcement learning algorithm according to which an **optimal policy** is searched by iteratively 
interacting with an environment and updating an **action value function** (Q-function). 

The agent decides its actions depending on its state (or state observation) by applying a **policy**. The pourpose of 
reinforcement learning is of learning an optimal policy, i.e. a plolicy which maximizes the 
**expected cumulative reward** collected by the agent.

The **action value function**, associated to a given policy **\pi**, represents the **expected _discounted_ cumulative 
reward** which will be collected by starting from each given state **S**, and taking each possible action **A**, then 
continuing to interact with the environment by applaying policy **\pi** until the end of an episode (or to the end of time
in a non-episodic environment). The action value function is also called **Q-function**

**Q-learning** is a reinforcement learning algorithm according to which an agent updates its representation of the 
Q-function after every action. 

An agent in state **S_{n}** will take action **A** selected according to a policy of choice. The environment will return
reward **R** and move the agent to state **S_{n+1}**. 

According to **Q-learning**, as a result of this action and state transition, the agent will be able to formulate a new 
estimation for the action value (Q-function value) of state S_{n} and action A.  
This new estimation is the sum 2 elements: 
- the current collected reward **R**
- the future estimated reward **G** which can be collected using an optimal policy to interact with the environment 

The future cumulative reward **G** is estimated form the current estimated Q-function as the highest action value among 
the set of possible actions at state **S_{n+1}**

The Q-learning algorithm gradually updates the extimated action values for states and actions **Q{S_{n}, A}** with this 
new immediate reward **R** and the estimation of the future cumulative rewards **G**.

In the case of **finite discrete state and action spaces**, the Q-function in Q-Learning can be easily represented with a 
table associating to each State-Action pair an estimated cumulative reward. 

In the case of a **continuous state space**, (noninear) **function approximation** can be used to represent the 
Q-function and to associate an expected cumulative reward to each possible state in the continuous state space and to each possible action.
A **Q-Network** implements this function approximation by means of a **(Deep) Neural Network**.


###Implementation

####Preliminary notes 

The main pourpose of this project is to understand get hands-on experience with the pyTorch library and with Deep 
Q-Networks and its variants. To get the most out of this assignment, I thought it better to start developing my 
DQNetwork from scratch, and to use the available DQN exercise solution only to identify possible problems at an avanced 
stage of the project.

For this reason, my implementation may deviate form the implementation proposed in the DQN solution and some of my 
implementations choices may be less effective or less intuitive than the solution design.

####Project structure


 - **solution.py** loads and uses the trained DQN agent solving the BananaCollection environment.
    The agent had been trained for less than 700 episodes using test_q_agent.py. 
 - **test_q_agent.py** is the main file used during development the QAgent class implemented in 
    q_agent.py and the file used for training the selected DQN agent.
 - **discrete_action_env.py**, **discrete_action_gym_env.py** and **banana_env.py**: wrappers to the gym and banana collection 
    environments, they have been developed so that the DQNet agent can be run both gym and BananaCollection environments 
    with the same code. 
    The interface closely matches the gym environment interface and is defined in discrete_action_env.py, while 
    discrete_action_gym_env.py and banana_env.py are 2 implementations.
    A related simple print based test file is provided.
 - **q_agent.py**: QAgent class implementing the Deep Q-Network algorithm. Uses pytorch_base_network.py for its neural 
    network and experience_replayer.py for its experience buffer. 
    The QAgent can be configured to use the original algorithm in [] or the double QNetwork algorithm in []. 
    The QAgent also exposes a method to save its internal state and configuration to file.
 - **pytorch_base_network.py**: PytorchBaseNetwork neural network class used by  QAgent. The hidden linear 
    configuration can be specified as a list or tuple of integers as an input parameter, and dropout can be optionally
    activated. The class provides a method for saving its weights and configuration to file, and its constructor can 
    load configuration and trained weights from a saved file. A related test file using unittest module is also 
    provided.
 - **experience_replayer.py**: ExperienceReplayer class used as experience buffer by QAgent. Experience objects
    or any other object can be stored in a fixed element size memory. The oldest entries are removed where maximum capacity
    is reached, and entries can be randomly sampled from memory. Optionally, a priority value can be associated to 
    each entry. In this case experiences will be extracted with a probability which is epsilon-proportional to their 
    priority. A simple test file is also provided.
 - **param_optim.py** is a random parameter optimization script used to tests several parameter configurations.

####Modification: Delayer network

Following the description of [], a local network is trained and used for interacting with the environment, while a 
target network is used as a reference in training, for cumputing the action values used for updating the 
action value function modelled by the local network. The target network is not directly trained, but it is instead 
updated with the weights of the local networks at regular intervals.
The target and the local network used by the agents are implemented by the PytorchBaseNetwork class in 
pytorch_base_network.py. The sequence of linear hidden layers used by the network can be specified in a list, and the 
network supports dropout. The PytorchBaseNetwork class also provides a method for saving its structure and weights to a 
file.

As an implementation choice, the proposed solution uses a soft update strategy instead of periodically copying all
weights of the local network to the target network. 
While this strategy seems reasonable and effective, it must be observed with this approach the target network are not 
only delayed but also temporally smoothed with respect to the local network weights: these smothed weights have never 
been used  by the local network and could in principle not be effective.

The approach suggested in [], of simply periodically replacing weights, also rise some perplexities. If a target
network with different weights is indeed needed to [**####**], then we might prefer this condition never to happen, 
while instead we will have identical local and target networks at regular intervals. This problem is indeed addressed by 
soft update of target network weights.

In order to have a target network which is always distinct from the local network, while still making sure that the 
weights of the target network are indeed past weights of the local network, an additional delayer 
network has been implemented. This network act as a distancing buffer between the local network and the target network.
The delayer network can be enabled via a parameter in the QAgent constructor.
I do not claim this modification to provide any significant improvement in performance, I have mainly implemented it as 
an exercise.

####Future Work

The project has developed with the idea of being able to extend it in the future.
 - The ExperienceReplayer class supports priority values and priority sampling, enabling to implement prioritized 
 experience replay in the future
 - The PytorchBaseNetwork class supports dropout, linear layers and a stub for the addition of convolutional layers.
 - The DiscreteActionEnv interface class eases the development of environment wrappers, so that the new envorimnents 
  can be easily tested.
  
