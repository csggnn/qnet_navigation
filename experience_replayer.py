from collections import deque

import numpy as np


class ExperienceReplayer:
    """ data storage for implementing (priorityzed) Experience replay """

    def __init__(self, max_length, default_prio=None, epsilon=0.1):
        """ Initialize an ExperienceReplayer object with or without priority

        :param max_length: size of the ExperienceReplayer buffer
        :param default_prio: default priority value for elements for which priority is not specified. when set to None
                    priority will not be used
        :param epsilon: between 0 and 1. Measures the weight of uniform sampling in drawing samples from memory
                    with epsilon = 1 priorities are ignored, with epsilon = 0 samples are drawn with probabilities
                    proportional to their priority value.
        """
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)
        self.def_prio = default_prio
        self.use_prio = (default_prio is not None)
        if self.use_prio:
            self.def_prio = float(self.def_prio)
        self.prio_epsilon = epsilon
        self.prio_memory = None
        self.norm_prio_array = np.array(float)
        self.norm_prio = None
        if self.use_prio:
            # self.prio_memory values are 1 to 1 associated with self.memory values
            self.prio_memory = deque(maxlen=max_length)
            self.norm_prio = False

    def store(self, sample, prio=None):
        """ store a new sample in memory (and forget the oldest sample if memory is full)

        :param sample: sample to be stored in memory
        :param prio: priority for the sample (positive value) if priority is used, ignored otherwise
        """
        self.memory.append(sample)
        if self.use_prio:
            self.norm_prio = False
            if prio is None:
                prio = self.def_prio
            self.prio_memory.append(float(prio))
        else:
            if prio is not None:
                print(
                    "ExperienceReplayer warning: priority value specified but current ExperienceReplayer object has not been initialsed to use priority, ignoring.")

    def draw(self, number):
        """ draw a "number" of samples from memory

        Samples are drawn with no duplicates (as long as duplicates have not been stored) and  either with uniform
        probability or using priority depending on initialization.
        If the memory does not store a sufficient number of samples, no sample will be drawn and this method will return
        None

        :param number: the number of samples to be drawn from memory
        :return: list of samples or None
        """
        if number > len(self.memory):
            return None
        if self.use_prio:
            if self.norm_prio is False:
                self.compute_norm_prio_array()
            return [self.memory[el] for el in
                    np.random.choice(len(self.memory), size=number, replace=False, p=self.norm_prio_array)]
        else:
            return [self.memory[el] for el in np.random.choice(len(self.memory), size=number, replace=False)]

    def compute_norm_prio_array(self):
        """ compute an array of probability values from the priority array

        as priorities for sample can be specified with any value, a function is needed to convert this priority values
        into probability values. Negative priority values should not exist and will be assumed 0.
        Each sample will take a probability value which is the sum of a proportional component (weighting prio_epsilon)
        and a uniform probability component.
        """
        self.norm_prio_array = np.array(self.prio_memory)
        self.norm_prio_array[self.norm_prio_array < 0] = 0
        self.norm_prio_array = self.norm_prio_array * (1 - self.prio_epsilon) / (
            sum(self.norm_prio_array)) + self.prio_epsilon / len(self.norm_prio_array)
        self.norm_prio = True
        return self.norm_prio_array
