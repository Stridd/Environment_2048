from parameters import Parameters
import numpy as np 

class Memory():
    def __init__(self):
        
        self.next_slot = 0
        self.max_size = Parameters.max_memory_size

        self.experiences = []

    def get_size(self):
        return len(self.experiences)

    def store_experience(self, experience):

        if len(self.experiences) < self.max_size:
            self.experiences.append(None)

        self.experiences[self.next_slot] = experience 

        self.next_slot = (self.next_slot + 1) % self.max_size

    def sample_experiences(self, batch_size):
        indexes = np.random.choice(self.next_slot, size = batch_size, replace = False)

        experiences = [self.experiences[index] for index in indexes]

        return experiences