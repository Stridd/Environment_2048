import numpy as np
import torch
import random
from pathlib import Path
from datetime import datetime

from .reinforce_policy import Reinforce_Policy

from Environment_2048 import Environment_2048

from history      import History
from data_helper  import Data_Helper
from loggers      import REINFORCELogger as Logger
from utilities    import Utility, RewardUtility, PreprocessingUtility,OptimizerUtility
from parameters   import REINFORCEParameters as params
from plotter      import Plotter

class Reinforce_Agent():
    def __init__(self):

        self.game =  Environment_2048(params.BOARD_SIZE)
     
        if params.SEED is not None:
            self.set_seed(params.SEED)

        # Also reset the game to take into account the new seed
        self.game.resetGame()
        
        self.time_of_experiment = Utility.get_time_of_experiment()
        
        self.logger = Logger(self.time_of_experiment, params)

        # Use the logger time of experiment to save figures to corresponding folder
        self.plotter = Plotter(self.time_of_experiment, params.PLOT_FOLDER_NAME)
        
        self.history = History()
        self.data_helper = Data_Helper()

        self.policy = Reinforce_Policy(self.history).to(params.DEVICE)
        self.policy.train()

        OptimizerUtility.set_params(params)
        self.optimizer = OptimizerUtility.get_optimizer_for_parameters(self.policy.parameters())

        self.trained_for = 0
        self.gamma = params.GAMMA

        if params.LOAD_MODEL == True:
            self.load_model_from(params.MODEL_PATH)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.game.setSeed(seed)
        
    def learn(self):
    
        episode_length = len(self.policy.rewards)
        returns = np.empty(episode_length, dtype = np.float32)

        eps = np.finfo(np.float32).eps.item()

        # Initialize this for better processing of future rewards
        future_returns = 0.0 

        for t in reversed(range(episode_length)):
            future_returns = self.policy.rewards[t] + self.gamma * future_returns
            returns[t] = future_returns

        normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

        normalized_returns = torch.tensor(normalized_returns).to(params.DEVICE)
        log_probabilities = torch.stack(self.policy.log_probablities).flatten().to(params.DEVICE)

        loss = -log_probabilities * normalized_returns
        # Loss needs to be an item, not an tensor.
        loss = torch.sum(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):

        for episode in range(params.EPISODES):
            self.play_until_end_of_game()
            
            # Keep this assignment here to increase clarity(PLAY GAME -> LEARN -> STORE DATA)
            self.data_helper.loss = self.learn()

            self.store_and_write_data()
            self.clean_up_episode_history()

        self.logger.save_parameters_to_json(params)
        self.save_model()
        self.save_obtained_cells(self.history)

    def play_until_end_of_game(self):
        
            game_is_done = False

            self.game.resetGame()

            while not game_is_done:

                game_board = self.game.getBoard()

                # Need available actions to 0-out the probabilities of the invalid moves
                # Also to check if the game is finished
                available_actions = self.game.getAvailableMoves(game_board, len(game_board))

                if len(available_actions) == 0:
                    self.game.setFinishedIfNoActionIsAvailable()
                else:
                    self.data_helper.game_board        = game_board
                    self.data_helper.available_actions = available_actions

                    self.perform_action_and_store_data()

                game_is_done = self.game.isFinished()
    
    def perform_action_and_store_data(self):
        reward = 0

        # Apply preprocessing and other operations
        state = PreprocessingUtility.transform_board_into_state(self.data_helper.game_board)

        # Need available_actions to rule out invalid actions
        action = self.policy.get_action(state, self.data_helper.available_actions)

        self.game.takeAction(action)

        reward = RewardUtility.get_reward(params.REWARD_FUNCTION, self.game.getMergedCellsAfterMove())
        # Store min and max reward for statistics
        self.data_helper.store_min_max_reward(reward)

        self.policy.store_reward(reward)

        # Store the game board to check clearly the state of the game
        self.history.store_state_action_reward_for_current_episode([self.data_helper.game_board, action, reward])

        # Increase the steps taken to see episode length
        self.data_helper.steps += 1
    
    def store_and_write_data(self):
        self.data_helper.game_board = self.game.getBoard()
        self.data_helper.total_reward = np.sum(self.policy.rewards)

        self.data_helper.store_max_cell_statistics()

        self.history.add_data_helper_info(self.data_helper)
        
        self.logger.write_data_to_logs_using_history(self.history)

    def clean_up_episode_history(self):
        self.logger.close_log_for_current_episode()

        self.data_helper.clear_current_data()
        self.history.clear_current_episode_data()

        self.history.increment_episode()
        self.trained_for += 1

        if self.history.current_episode < params.EPISODES:
            self.logger.open_new_log_for_current_episode(self.history)

        self.policy.reset_policy()

    def plot_statistics_to_files(self):
        self.plotter.generate_and_save_plots_from_history(self.history)

    def save_model(self):
        
        current_directory = Utility.get_absolute_path_from_file_name(__file__)

        logs_folder = str(Path(current_directory).parents[1]) + '\\' + params.LOG_FOLDER_NAME
        path = logs_folder + '\\' + self.time_of_experiment   + '\\' + params.MODEL_NAME

        torch.save({
            'trained_for'         : self.trained_for,
            'model_state_dict'    : self.policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

    def save_obtained_cells(self, agent_history):
        self.logger.write_obtained_cells(agent_history)

    def load_model_from(self, path):
        checkpoint = torch.load(path)
        self.policy.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trained_for = checkpoint['trained_for']
