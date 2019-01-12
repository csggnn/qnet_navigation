from collections import deque
import numpy as np


class ExperienceReplayer:
    """ data storage for implementing (priorityzed) Experience replay """

    def __init__(self, max_length, default_prio = None, epsilon=0.1):
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)
        self.def_prio =  default_prio
        self.use_prio = (default_prio is not None)
        if  self.use_prio:
            self.def_prio =  float(self.def_prio)
        self.prio_epsilon=epsilon
        self.prio_memory = None
        self.norm_prio_array = np.array(float)
        self.norm_prio=None
        if self.use_prio:
            #self.prio_memory values are 1 to 1 associated with self.memory values
            self.prio_memory = deque(maxlen=max_length)
            self.norm_prio=False

    def store(self, value, prio = None):
        self.memory.append(value)
        if self.use_prio:
            self.norm_prio = False
            if prio is None:
                prio = self.def_prio
            self.prio_memory.append(float(prio))

    def draw(self, number):
        if number>len(self.memory):
            return None
        if self.use_prio:
            if self.norm_prio is False:
                self.compute_norm_prio_array()
            return np.random.choice(self.memory, number, False, self.norm_prio_array)
        else:
            return np.random.choice(self.memory, number, False)

    def compute_norm_prio_array(self):
        self.norm_prio_array = np.array(self.prio_memory)
        self.norm_prio_array[self.norm_prio_array < 0] = 0
        self.norm_prio_array = self.norm_prio_array *(1-self.prio_epsilon) / (sum(self.norm_prio_array))+self.prio_epsilon/len(self.norm_prio_array)
        self.norm_prio = True
        return self.norm_prio_array






