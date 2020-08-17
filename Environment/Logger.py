import os 
import Parameters

class Logger():
    def __init__(self):
        
        logs_folder_name = Parameters.logs_folder_name
        logs_folder_path = logs_path + '\\' + logs_folder_name

        if not os.path.isdir(logs_folder_path):
            os.mkdir(logs_folder_path)

        path_to_folder = logs_folder_path + '\\'

        episode_info_path = path_to_folder + Parameters.episode_data_file_name
        general_info_path = path_to_folder + Parameters.general_info_file_name

        self.episode_info_file = open(episode_info_path, 'w+')
        self.general_info_file = open(general_info_path, 'w+')

    def log_info(self, message):
        self.general_info_file.write(message)
        self.general_info_file.write('\n')

    def write_statistics(self, agent_history):
        losses          = agent_history.get_losses()
        rewards         = agent_history.get_episode_rewards()
        episode_lengths = agent_history.get_episode_lengths()
        min_rewards     = agent_history.get_min_rewards()
        max_rewards     = agent_history.get_max_rewards()
        max_cells       = agent_history.get_max_cells()
        max_cells_count = agent_history.get_max_cells_count()

        for i in range(len(losses)):
           self.episode_info_file.write('| Episode %5s | Lengths: %-5s | Loss: %-5.2f |Min Reward: %-5s|Max Reward: %-5s|Total Reward: %-5s|Max cell: %-5s|Max Cell Count: %-3s| ' \
                                         % (i, episode_lengths[i], losses[i], min_rewards[i], max_rewards[i], rewards[i], max_cells[i], max_cells_count[i]))
           self.episode_info_file.write('\n')

    def __del__(self):
        print('Closing files')
        self.episode_info_file.close()
        self.general_info_file.close()