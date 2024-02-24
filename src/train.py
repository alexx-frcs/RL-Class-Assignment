from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn 

device = torch.device("cpu")


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        return 0

    def save(self, path):
        pass

    def load(self):
        pass

class DQNAgent():
    def __init__(self):
        self.state_dim = env.observation_space.shape[0]
        self.n_action = env.action_space.n 
        self.nb_neurons=24
    

        DQN = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.n_action)).to(device)
