import numpy as np
import torch
import torch.nn.functional as F
import random


from .networks import Network, TargetNetwork
from pathlib import Path
from Environment_2048 import Environment_2048

from history            import DQNHistory as History
from data_helper        import Data_Helper
from loggers            import DQNLogger
from plotter            import DQNPlotter
from utilities          import Utility,RewardUtility,PreprocessingUtility,OptimizerUtility
from memory.memory      import Memory
from parameters         import DDQNParameters as PARAM

class DDQN():
    def __init__(self):

        self.net           = Network().to(PARAM.DEVICE)
        self.target_net    = TargetNetwork().to(PARAM.DEVICE)

        self.replay_memory = Memory(PARAM.MEMORY_SIZE)
        self.epsilon       = PARAM.EPSILON

        self.game          = Environment_2048(PARAM.BOARD_SIZE)

        if PARAM.SEED is not None:
            self.set_seed(PARAM.SEED)

        self.game.resetGame()

        self.time_of_experiment = Utility.get_time_of_experiment()
        self.gamma              = PARAM.GAMMA

        self.logger = DQNLogger(self.time_of_experiment, PARAM)
        self.plotter = DQNPlotter(self.time_of_experiment,PARAM.PLOT_FOLDER_NAME)
        
        self.history = History()
        self.data_helper = Data_Helper()

        OptimizerUtility.set_params(PARAM)
        self.optimizer = OptimizerUtility.get_optimizer_for_parameters(self.net.parameters())

        self.trained_for = 0
        self.max_cell_achieved = -1

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.game.setSeed(seed)

    def train(self):

        for episode in range(PARAM.EPISODES):
            print('Processing episode: {}'.format(episode))
            self.play_until_end_of_game()
            self.write_data()
            self.clean_up_episode_history()

            if episode % PARAM.UPDATE_FREQUENCY == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            self.trained_for += 1
        
        self.logger.save_parameters_to_json(PARAM)
        self.save_model()
        self.save_obtained_cells(self.history)

    def play_until_end_of_game(self):
        game_is_done = False

        self.game.resetGame()

        available_actions = None

        while not game_is_done:

            game_board = self.game.getBoard()

            # Need available actions to 0-out the probabilities of the invalid moves
            # Also to check if the game is finished
            if available_actions == None:
                available_actions = self.game.getAvailableMoves(game_board, len(game_board))

            state = PreprocessingUtility.transform_board_into_state(game_board)

            action = self.get_action(state, available_actions)

            self.game.takeAction(action)

            reward = RewardUtility.get_reward(PARAM.REWARD_FUNCTION, self.game.getMergedCellsAfterMove()) 
            # See here for formula: https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
            reward = (((reward - 0) * (1 - (-1))) / (512 - 0)) + -1

            self.history.store_state_action_reward_for_current_episode((game_board, action, reward))
            self.data_helper.store_min_max_reward(reward)

            next_game_board = self.game.getBoard()
            next_state = PreprocessingUtility.transform_board_into_state(next_game_board)

            available_actions = self.game.getAvailableMoves(next_game_board, len(next_game_board))

            if len(available_actions) == 0:
                self.game.setFinishedIfNoActionIsAvailable()

            game_is_done = self.game.isFinished()

            self.replay_memory.store_experience( (state,action,reward,next_state,game_is_done) ) 

            loss = self.learn()
            loss = loss.item() if loss is not None else 'None'

            self.history.store_loss(loss)

            self.data_helper.steps += 1
        
        self.store_data_in_history()

    def get_action(self, state, available_actions):
        # Sample a random
        # If less than epsilon, then choose randomly, otherwise, use network to select action
        sample = np.random.random()
        action = None

        if sample < self.epsilon:
            action = np.random.choice(available_actions)
            self.history.store_action_type('exploration')
            self.history.store_net_q_values([np.float32('-Inf') for i in range(4)])
        else:
            self.history.store_action_type('exploitation')
            state = torch.tensor(state, device = PARAM.DEVICE)

            # Set all actions as invalid (with return -Inf)
            q_values = np.ones(shape = (4,), dtype = np.float32) * np.float32('-Inf')

            predicted_values = self.net.forward(state)

            # Correct the values for the available actions
            predicted_q_values = predicted_values.cpu().detach()

            self.history.store_net_q_values(predicted_q_values.numpy())

            q_values[available_actions] = predicted_q_values[available_actions]

            # See link here: https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
            action = np.random.choice(np.where(q_values == q_values.max())[0])
        
        return action

    def learn(self):

        if self.replay_memory.get_size() < PARAM.BATCH_SIZE:
            return np.float32('-Inf')

        experiences = self.replay_memory.sample_experiences(PARAM.BATCH_SIZE)

        # Unpack the tuple into each separate list
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Squeeze because at the moment we have (X,1,16)
        states      = torch.tensor(states,       device = PARAM.DEVICE, dtype = torch.float32)
        actions     = torch.tensor(actions,      device = PARAM.DEVICE, dtype = torch.long)
        rewards     = torch.tensor(rewards,      device = PARAM.DEVICE, dtype = torch.float32)
        next_states = torch.tensor(next_states,  device = PARAM.DEVICE, dtype = torch.float32)
        dones       = torch.tensor(dones,        device = PARAM.DEVICE, dtype = torch.float32)

        # Predict the action values for the Q(s, a) for all a
        net_action_values = self.net(states) 
        # Only choose the actions that were actually taken
        q_values = net_action_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # Compute the same values for the next_states 
            next_state_q_values = self.net(next_states)

            # Use the eval network to predict the values
            target_q_values     = self.target_net(next_states)

        # Get the argmax for the next states (as in the formula)
        _, next_state_action_indices = torch.max(next_state_q_values, dim = 1)
        
        # Now use the values from the target_network to get the corresponding q-value
        max_next_q_preds = target_q_values.gather(-1, next_state_action_indices.unsqueeze(1)).squeeze()

        expected_q_value = rewards + self.gamma * max_next_q_preds * (1 - dones)

        # TO-DO: Replace loss with more easily modifiable function
        loss = F.smooth_l1_loss(q_values, expected_q_value.data)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def store_data_in_history(self):
        self.data_helper.game_board = self.game.getBoard()
        self.data_helper.store_max_cell_statistics()

        self.history.store_episode_length(self.data_helper.steps)
        self.history.store_total_reward(self.history.rewards_current_episode)
        self.history.store_average_reward(self.history.rewards_current_episode)
        self.history.store_max_cell(self.data_helper.max_cell)
        self.history.store_max_cell_count(self.data_helper.max_cell_count)
        self.history.store_min_reward(self.data_helper.min_reward)
        self.history.store_max_reward(self.data_helper.max_reward)

    def write_data(self):
        self.logger.write_data_for_current_episode_using_history(self.history)
        self.logger.write_experiment_info_using_history(self.history)

        self.logger.close_log_for_current_episode()

        self.history.increment_episode()
        if self.history.current_episode < PARAM.EPISODES:
            self.logger.open_new_log_for_current_episode(self.history)

    def clean_up_episode_history(self):
        self.history.clear_current_episode_data()
        self.data_helper.clear_current_data()

    def save_model(self):
        
        current_directory = Utility.get_absolute_path_from_file_name(__file__)

        logs_folder = str(Path(current_directory).parents[1]) + '\\' + PARAM.LOG_FOLDER_NAME
        path = logs_folder + '\\' + self.time_of_experiment   + '\\' + PARAM.MODEL_NAME

        torch.save({
            'trained_for'            : self.trained_for,
            'q_values_net_dict'      : self.net.model.state_dict(),
            'target_net_dict'        : self.target_net.model.state_dict(),
            'optimizer_state_dict'   : self.optimizer.state_dict(),
            }, path)

    def save_obtained_cells(self, agent_history):
        self.logger.write_obtained_cells(agent_history)

    def plot_statistics_to_files(self):
        self.plotter.generate_and_save_plots_from_history(self.history)