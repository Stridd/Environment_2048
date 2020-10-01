from Parameters import Parameters
from Utility import Utility
from DataUtility import DataUtility
import glob
import os
import json
import pandas as pd
import numpy as np

from datetime import datetime

class Logger():
    def __init__(self, logs_path, time_of_experiment):
        
        logs_folder_name = Parameters.logs_folder_name
        logs_folder_path = logs_path + '\\' + logs_folder_name

        Utility.make_folder_if_not_exist(logs_folder_path)

        self.path_to_folder = logs_folder_path + '\\' + time_of_experiment + '\\'

        Utility.make_folder_if_not_exist(self.path_to_folder)

        self.path_to_episode_logs = self.path_to_folder + '\\' + 'episodic_logs' + '\\'

        Utility.make_folder_if_not_exist(self.path_to_episode_logs)

        # When it's initialized the first episode is the 0 episode
        self.log_name_for_current_episode = self.path_to_episode_logs + 'episode_0' + '.csv'
        self.csv_for_current_episode = pd.DataFrame(columns=['State','Reward','Action','Entropy'])

        self.experiment_info_path = self.path_to_folder + Parameters.experiment_data_file_name

        self.build_data_per_episode_format()
        self.build_experiment_info_format()

    def build_data_per_episode_format(self):

        self.data_per_episode_format = '| State: %80s |'
        self.data_per_episode_format += ' Output: %40s|'
        self.data_per_episode_format += 'Entropy: %10.4f|'
        self.data_per_episode_format += 'Action:  %-5s|'
        self.data_per_episode_format += 'Reward:  %-10.2f|'

    def build_experiment_info_format(self):
        self.experiment_info_format = '| Episode %5s |'
        self.experiment_info_format += 'Length: %-10s|'
        self.experiment_info_format += 'Loss: %10.2f|'
        self.experiment_info_format += 'Min Reward: %-10.2f|'
        self.experiment_info_format += 'Max Reward: %-10.2f|'
        self.experiment_info_format += 'Total Reward: %-10.2f|'
        self.experiment_info_format += 'Max cell: %-10s|'
        self.experiment_info_format += 'Max cell count: %-3s|'

    def write_data_to_logs_using_history(self, agent_history):
        self.write_experiment_info_using_history(agent_history)
        self.write_data_for_current_episode_using_history(agent_history)
    
    def open_new_log_for_current_episode(self, agent_history):
        current_episode = agent_history.current_episode
        self.log_name_for_current_episode = self.path_to_episode_logs + 'episode_' + str(current_episode) + '.csv'
        self.csv_for_current_episode = pd.DataFrame(columns=['State','Reward','Action','Entropy'])

    def close_log_for_current_episode(self):
        self.csv_for_current_episode.to_csv(self.log_name_for_current_episode)

    def write_experiment_info_using_history(self, agent_history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count
        current_episode = agent_history.current_episode

        experiment_info_file = open(self.experiment_info_path, 'a+')

        experiment_info_file.write(self.experiment_info_format
                                         % (current_episode, 
                                            episode_lengths[current_episode], 
                                            losses[current_episode], 
                                            min_rewards[current_episode], 
                                            max_rewards[current_episode], 
                                            rewards[current_episode],
                                            max_cells[current_episode], 
                                            max_cells_count[current_episode]))
           
        experiment_info_file.write('\n')
        experiment_info_file.close()

    def pretty_write_states_in_csv(self, states_per_episode):

        for i in range(len(states_per_episode)):
            self.csv_for_current_episode.at[i + 1,'State'] = ""
            number_of_rows = len(states_per_episode[i])
            for j in range(number_of_rows): 
                self.csv_for_current_episode.at[i+1,'State'] += str(states_per_episode[i][j])
                if j != number_of_rows - 1:
                    self.csv_for_current_episode.at[i+1,'State'] += '\r\n'

    def write_data_for_current_episode_using_history(self, agent_history):

        state_evolution_current_episode         = agent_history.state_evolution_current_episode
        rewards_current_episode                 = agent_history.rewards_current_episode
        actions_current_episode                 = agent_history.actions_current_episode
        network_output                          = agent_history.network_output
        entropy                                 = agent_history.entropy
        current_episode                         = agent_history.current_episode
 
        self.pretty_write_states_in_csv(state_evolution_current_episode)
        self.csv_for_current_episode.loc[:, 'Reward'] = rewards_current_episode
        self.csv_for_current_episode.loc[:, 'Action'] = actions_current_episode
        #self.csv_for_current_episode.loc[:, 3] = network_output
        self.csv_for_current_episode.loc[:, 'Entropy'] = entropy

    def save_parameters_to_json(self):
        parameters_file_path = self.path_to_folder + 'Parameters.json'
        json_content = DataUtility.get_parameters_class_as_json()

        with open(parameters_file_path, 'w+') as fp:
            json.dump(json_content, fp,  sort_keys=True, indent=4)