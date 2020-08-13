from Reinforce_Policy import Reinforce_Policy
from Environment_2048 import Environment_2048
import torch.optim as optim

class Reinforce_Agent():
    def __init__(self, input_size, output_size, gamma, optimizer):
        self.policy = Reinforce_Policy(input_size, output_size)
        self.gamma  = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.game =  Environment_2048(4)

    def train(self, optimizer):
        episode_length = len(self.policy.rewards)
        returns = np.empty(episode_length, dtype = np.float32)

        # Initialize this for better processing of future rewards
        future_returns = 0.0 

        for t in reversed(range(episode_length)):
            future_returns = self.policy.rewards[t] + self.gamma * future_returns
            returns[t] = future_returns

        returns = torch.tensor(returns)
        log_probabilities = torch.stack(self.policy.log_probablities)

        loss = -log_probabilities * returns
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def learn(self, episodes):
        game_is_done = False

        while not game_is_done:

            action = self.game.sampleAction()
            if action == -1:
                self.game.setFinishedIfNoActionIsAvailable()
            else:
                self.game.takeAction(action)
            
            print(self.game.getMergedCellsAfterMove())
            game_is_done = self.game.isFinished()
    