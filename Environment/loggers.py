from utilities import Utility, DataUtility
from abc import ABC, abstractmethod

import os
import json
import numpy as np

from datetime import datetime

class Logger(ABC):
    def __init__(self, time_of_experiment,  parameters):
        
        self.create_folders_for_current_experiment(time_of_experiment, parameters)

        # When it's initialized the first episode is the 0 episode
        log_file_name = self.path_to_episode_logs + 'episode_' + '%s' + '.txt'
        self.log_for_current_episode = open(log_file_name % 0, 'a+')

        self.experiment_info_path = self.path_to_folder + parameters.EXPERIMENT_FILE

        self.write_headers_for_current_log()
        self.write_headers_for_episodic_log()

        self.write_separation_line_for_episodic_log()
        self.write_separation_line_for_current_log()

        self.build_experiment_info_format()
        self.build_data_per_episode_format()

        self.parameters_file_name     = parameters.PARAMETERS_FILE
        self.obtained_cells_file_name = parameters.OBTAINED_CELLS_FILE

    def create_folders_for_current_experiment(self, time_of_experiment, parameters):
        
        logger_location = Utility.get_absolute_path_from_file_name(__file__)

        logs_folder_path            = logger_location + '\\' + parameters.LOG_FOLDER_NAME

        self.path_to_folder         = logs_folder_path + '\\' + time_of_experiment + '\\'
        self.path_to_episode_logs   = self.path_to_folder + '\\' + 'episodic_logs' + '\\'

        for folder in [logs_folder_path, self.path_to_folder, self.path_to_episode_logs]:
            Utility.make_folder_if_not_exist(folder)

    @abstractmethod
    def build_experiment_info_format(self):
        pass

    @abstractmethod
    def build_data_per_episode_format(self):
        pass

    @abstractmethod 
    def write_headers_for_current_log(self):
        pass 
        
    @abstractmethod
    def write_separation_line_for_current_log(self):
        pass

    @abstractmethod
    def write_headers_for_episodic_log(self):
        pass

    @abstractmethod
    def write_separation_line_for_episodic_log(self):
        pass

    def write_data_to_logs_using_history(self, agent_history):
        self.write_experiment_info_using_history(agent_history)
        self.write_data_for_current_episode_using_history(agent_history)
    
    def open_new_log_for_current_episode(self, agent_history):
        current_episode = agent_history.current_episode
        log_file_name = self.path_to_episode_logs + 'episode_' + '%s' + '.txt'
        self.log_for_current_episode = open(log_file_name % current_episode, 'a+')

        self.write_headers_for_current_log()
        self.write_separation_line_for_current_log()

    def close_log_for_current_episode(self):
         self.log_for_current_episode.close()

    @abstractmethod
    def write_experiment_info_using_history(self, agent_history):
        pass 

    @abstractmethod
    def write_data_for_current_episode_using_history(self, agent_history):
        pass 

    def save_parameters_to_json(self, param):
        parameters_file_path = self.path_to_folder + self.parameters_file_name
        json_content = param.get_constants_as_json()

        with open(parameters_file_path, 'w+') as f:
            json.dump(json_content, f,  sort_keys = True, indent = 4)
    
    def write_obtained_cells(self, agent_history):
        cells_and_occurences = DataUtility.build_and_sort_max_cell_distribution_from_history(agent_history)

        cell_distribution_file_path = self.path_to_folder + self.obtained_cells_file_name
        with open(cell_distribution_file_path, 'w') as f:
            # Do not sort the keys as they are already sorted
            json.dump(cells_and_occurences, f,  sort_keys=False, indent =4)

class REINFORCELogger(Logger):
    def __init__(self, time_of_experiment, parameters):
        super().__init__(time_of_experiment, parameters)
        
    def build_experiment_info_format(self):
        self.experiment_info_format = '|{0:^11}|'
        self.experiment_info_format += '{1:^16}|'
        self.experiment_info_format += '{2:^14.4f}|'
        self.experiment_info_format += '{3:^12}|'
        self.experiment_info_format += '{4:^12}|'
        self.experiment_info_format += '{5:^14}|'
        self.experiment_info_format += '{6:^10}|'
        self.experiment_info_format += '{7:^7}|'

    def build_data_per_episode_format(self):
        self.data_per_episode_format = '|{0:^80}|{1:^40}|{2:^15.4f}|{3:^8}|{4:^11}|'

    def write_headers_for_current_log(self):
        # Also write the headers. Cannot use data_per_episode_format because it has a float value
        headers = ['state','network_output','entropy','action','rewards']
        headers_format = '|{0:^80}|{1:^40}|{2:^15}|{3:^8}|{4:^11}|'
        
        self.log_for_current_episode.write(headers_format.format(*headers))
        self.log_for_current_episode.write('\n')
        
    def write_separation_line_for_current_log(self):
        # Taken from the header format. The formula is:
        # Each of the numbers in the format string + the number of bars
        number_of_characters = 80 + 40 + 15 + 8 + 11 + 6
        self.log_for_current_episode.write('-' * number_of_characters)
        self.log_for_current_episode.write('\n')

    def write_headers_for_episodic_log(self):
        headers = ['episode','episode_length','loss','min_reward','max_reward','total_reward','max_cell','count']
        headers_format = '|{0:^11}|{1:^16}|{2:^14}|{3:^12}|{4:^12}|{5:^14}|{6:^10}|{7:^7}|'

        self.write_text_in_experiment_info(headers_format.format(*headers))

    def write_separation_line_for_episodic_log(self):

        number_of_characters = 11 + 16 + 14 + 12 + 12 + 14 + 10 + 7 + 9

        self.write_text_in_experiment_info('-' * number_of_characters)

    def write_experiment_info_using_history(self, agent_history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count
        current_episode = agent_history.current_episode

        data = self.experiment_info_format.format(current_episode, 
                                                episode_lengths[current_episode], 
                                                losses[current_episode], 
                                                min_rewards[current_episode], 
                                                max_rewards[current_episode], 
                                                rewards[current_episode],
                                                max_cells[current_episode], 
                                                max_cells_count[current_episode])
        self.write_text_in_experiment_info(data)

    def write_text_in_experiment_info(self, text):
        experiment_info_file = open(self.experiment_info_path, 'a+')
        experiment_info_file.write(text)
        experiment_info_file.write('\n')
        experiment_info_file.close()

    def write_data_for_current_episode_using_history(self, agent_history):

        state_evolution_current_episode         = agent_history.state_evolution_current_episode
        rewards_current_episode                 = agent_history.rewards_current_episode
        actions_current_episode                 = agent_history.actions_current_episode
        network_output                          = agent_history.network_output
        entropy                                 = agent_history.entropy
        current_episode                         = agent_history.current_episode
 
        network_output = np.array(network_output)

        for i in range(len(entropy)):
            text = self.data_per_episode_format.format(
                                            str(state_evolution_current_episode[i]),
                                            str(['%.3f' % output for output in network_output[i]]), 
                                            entropy[i], 
                                            actions_current_episode[i], 
                                            rewards_current_episode[i])
            self.log_for_current_episode.write(text)
            self.log_for_current_episode.write('\n')

class DQNLogger(Logger):
    def __init__(self, time_of_experiment, parameters):
        super().__init__(time_of_experiment, parameters)

    def build_experiment_info_format(self):
        self.experiment_info_format = '|{0:^11}|'
        self.experiment_info_format += '{1:^16}|'
        self.experiment_info_format += '{2:^14.2f}|'
        self.experiment_info_format += '{3:^16.2f}|'
        self.experiment_info_format += '{4:^12.2f}|'
        self.experiment_info_format += '{5:^14.2f}|'
        self.experiment_info_format += '{6:^10}|'
        self.experiment_info_format += '{7:^10}|'

    def build_data_per_episode_format(self):
        self.data_per_episode_format = '|{0:^80}|{1:^10}|{2:^14.2f}|{3:^14}|{4:^50}|{5:^16.2f}|{6:^18.4f}|'

    def write_headers_for_current_log(self):
        # Also write the headers. Cannot use data_per_episode_format because it has a float value
        headers = ['state','action','reward','action_type','network_output','loss','exploration_rate']
        headers_format = '|{0:^80}|{1:^10}|{2:^14}|{3:^14}|{4:^50}|{5:^16}|{6:^18}|'
        
        self.log_for_current_episode.write(headers_format.format(*headers))
        self.log_for_current_episode.write('\n')
        
    def write_separation_line_for_current_log(self):
        # Taken from the header format. The formula is:
        # Each of the numbers in the format string + the number of bars
        number_of_characters = 80 + 10 + 10 + 10 + 40 + 14 + 7
        self.log_for_current_episode.write('-' * number_of_characters)
        self.log_for_current_episode.write('\n')

    def write_headers_for_episodic_log(self):
        headers = ['episode','episode_length','total_reward','average_reward','min_reward','max_reward','max_cell','count']
        headers_format = '|{0:^11}|{1:^16}|{2:^14}|{3:^16}|{4:^12}|{5:^14}|{6:^10}|{7:^10}|'

        self.write_text_in_experiment_info(headers_format.format(*headers))

    def write_separation_line_for_episodic_log(self):

        number_of_characters = 11 + 16 + 14 + 12 + 12 + 14 + 10 + 7 + 9

        self.write_text_in_experiment_info('-' * number_of_characters)

    def write_experiment_info_using_history(self, agent_history):

        current_episode = agent_history.current_episode
        episode_lengths = agent_history.episode_lengths
        total_rewards   = agent_history.episode_rewards
        average_rewards = agent_history.average_rewards
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count
        
        data = self.experiment_info_format.format(current_episode, 
                                                episode_lengths[current_episode], 
                                                total_rewards[current_episode],
                                                average_rewards[current_episode], 
                                                min_rewards[current_episode], 
                                                max_rewards[current_episode],
                                                max_cells[current_episode], 
                                                max_cells_count[current_episode])
        self.write_text_in_experiment_info(data)

    def write_text_in_experiment_info(self, text):
        experiment_info_file = open(self.experiment_info_path, 'a+')
        experiment_info_file.write(text)
        experiment_info_file.write('\n')
        experiment_info_file.close()

    def write_data_for_current_episode_using_history(self, agent_history):

        state_evolution_current_episode         = agent_history.state_evolution_current_episode
        rewards_current_episode                 = agent_history.rewards_current_episode
        actions_current_episode                 = agent_history.actions_current_episode
        network_output                          = agent_history.network_outputs
        action_type                             = agent_history.action_type
        loss                                    = agent_history.losses
        exploration_rate                        = agent_history.exploration_rate

        network_output = np.array(network_output)

        for i in range(len(loss)):
            text = self.data_per_episode_format.format(
                                            str(state_evolution_current_episode[i]),
                                            actions_current_episode[i],
                                            rewards_current_episode[i],
                                            action_type[i],
                                            str(['%.3f' % output for output in network_output[i]]), 
                                            loss[i],
                                            exploration_rate[i]
                                            )
            self.log_for_current_episode.write(text)
            self.log_for_current_episode.write('\n')