from Parameters import Parameters
import glob
import os 

class Logger():
    def __init__(self, logs_path):
        
        logs_folder_name = Parameters.logs_folder_name
        logs_folder_path = logs_path + '\\' + logs_folder_name

        if not os.path.isdir(logs_folder_path):
            os.mkdir(logs_folder_path)
        else:
            old_logs = glob.glob(logs_folder_path + '\\' + '.*')
            for f in old_logs:
                os.remove(f)

        self.path_to_folder = logs_folder_path + '\\'

        self.episode_info_path = self.path_to_folder + Parameters.episode_data_file_name

        self.build_data_per_episode_format()
        self.build_episode_info_format()

    def build_data_per_episode_format(self):

        self.data_per_episode_format = '| State: %60s |'
        self.data_per_episode_format += ' Output: %-40s|'
        self.data_per_episode_format += 'Entropy: %-5.4f|'
        self.data_per_episode_format += 'Action:  %-5s|'
        self.data_per_episode_format += 'Reward:  %-10s|'

    def build_episode_info_format(self):
        self.episode_info_format = '| Episode %5s |'
        self.episode_info_format += 'Length: %-10s|'
        self.episode_info_format += 'Loss: %-5.2f|'
        self.episode_info_format += 'Min Reward: %-10s|'
        self.episode_info_format += 'Max Reward: %-10s|'
        self.episode_info_format += 'Total Reward: %-10s|'
        self.episode_info_format += 'Max cell: %-10s|'
        self.episode_info_format += 'Max Cell Count: %-3s|'

    def write_episodic_info_using_history(self, agent_history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count
        current_episode = agent_history.current_episode

        episode_info_file = open(self.episode_info_path, 'a+')

        episode_info_file.write(self.episode_info_format 
                                         % (current_episode, 
                                            episode_lengths[current_episode], 
                                            losses[current_episode], 
                                            min_rewards[current_episode], 
                                            max_rewards[current_episode], 
                                            rewards[current_episode],
                                            max_cells[current_episode], 
                                            max_cells_count[current_episode]))
           
        episode_info_file.write('\n')
        episode_info_file.close()

    def write_data_per_episode_using_history(self, agent_history):

        state_evolution_current_episode         = agent_history.state_evolution_current_episode
        rewards_current_episode                 = agent_history.rewards_current_episode
        actions_current_episode                 = agent_history.actions_current_episode
        network_output                          = agent_history.network_output
        entropy                                 = agent_history.entropy
        current_episode                         = agent_history.current_episode

        file_name = self.path_to_folder + 'episode_' + '%s' + '.txt'
        with open(file_name % current_episode, 'a+') as episode_file:
            episode_file.write(self.data_per_episode_format
                                    % (state_evolution_current_episode[-1],
                                        ['%.3f' % output for output in network_output[-1]], 
                                        entropy[-1], 
                                        actions_current_episode[-1], 
                                        rewards_current_episode[-1]))
            episode_file.write('\n')

                   