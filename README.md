# QNet Navigation 

**QNet Navigation** is a PyTorch project implementing a reinforcement learning 
agent navigating in a *Banana Collection Environment*. 
In a Banana Collection Environemnt, reinforcement learning agent must be trained to 
collect yellow bananas and avoid blue bananas. You can see a video of a trained 
agent moving in a banana collection environment [here](https://youtu.be/L3VQbwDsEB4).

 
The agent in this project implements [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and was developed as
 a solution to  the navigation project for the 
[Udacity Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).


## Installation

In order to run the code in this project, you will need a python3.6 environment with the 
following packages:
- numpy
- matplotlib
- pickle
- torch

You will also need to install the Unity Banana Collection Environment according to your platform
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Win_32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Win_64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


If you want to run *param_optim_analysis.py*, also install the *pandas* package

The project also allows for training a DQN agent for 2 extra **gym** enviroments. 
- More information on **gym** and installation instructions at this [link](https://github.com/openai/gym)
- If you decide not to install **gym**, just comment its import in **tarin_DQN_agent.py**

## Usage

Use **run_trained_DQN_agent.py** to load a trained DQN agent solving the BananaCollection environment.

Use **train_DQN_agent.py** to train a new agent in a Banana Collection Environment. Other environments are also 
supported and can be used for testing and training.

Both files rely on **q_agent.py** for the DQN implementation. More details on the file structure, implementation choices
and parameters in **Report.md**