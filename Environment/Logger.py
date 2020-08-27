import os 
from Parameters import Parameters

class Logger():
    def __init__(self, logs_path):
        
        logs_folder_name = Parameters.logs_folder_name
        logs_folder_path = logs_path + '\\' + logs_folder_name

        if not os.path.isdir(logs_folder_path):
            os.mkdir(logs_folder_path)

        path_to_folder = logs_folder_path + '\\'

        episode_info_path = path_to_folder + Parameters.episode_data_file_name
        general_info_path = path_to_folder + Parameters.general_data_file_name

        self.episode_info_file = open(episode_info_path, 'w+')
        self.general_info_file = open(general_info_path, 'w+')

    def log_info(self, message):
        self.general_info_file.write(message)
        self.general_info_file.write('\n')

    def write_statistics(self, agent_history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count

        for i in range(len(losses)):
           self.episode_info_file.write('| Episode %5s | Length: %-10s | Loss: %-5.2f |Min Reward: %-10s|Max Reward: %-10s|Total Reward: %-10s|Max cell: %-10s|Max Cell Count: %-3s| ' \
                                         % (i, episode_lengths[i], losses[i], min_rewards[i], max_rewards[i], rewards[i], max_cells[i], max_cells_count[i]))
           self.episode_info_file.write('\n')

        reward_evolution_per_episode = agent_history.rewards_on_action_per_episode
        action_per_episode           = agent_history.actions_taken_per_episode
        state_evolution_per_episode  = agent_history.state_evolution_per_episode
        network_output               = agent_history.network_output
        entropy                      = agent_history.entropy_per_episode

        for i in range(len(reward_evolution_per_episode)):
            for j in range(len(reward_evolution_per_episode[i])):
                
                self.general_info_file.write('| Output: %50s | Entropy: %10.4f | Action: %-10s | Reward: %-10s |'  \
                                            % (network_output[i][j], entropy[i][j], action_per_episode[i][j], reward_evolution_per_episode[i][j]))
                self.general_info_file.write('\n')

            self.general_info_file.write('*' * 100)
            self.general_info_file.write('\n')

    def __del__(self):
        print('Closing files')
        self.episode_info_file.close()
        self.general_info_file.close()