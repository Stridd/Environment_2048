from Reinforce_Policy import Reinforce_Policy
from torch.distributions import Categorical

class Reinforce_Agent():
    def __init__(self, input_size, output_size, alpha, gamma):
        self.policy = Reinforce_Policy(input_size, output_size)
        self.alpha  = alpha
        self.gamma  = gamma

    def sample_action(self, state):
        probabilities = self.policy.forward(state)
        categorical = Categorical(logits = probabilities)
        sample = categorical.sample()
