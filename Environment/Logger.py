import os 
from Parameters import Parameters

class Logger():
    def __init__(self, logs_path):
        
        logs_folder_name = Parameters.logs_folder_name
        logs_folder_path = logs_path + '\\' + logs_folder_name

        if not os.path.isdir(logs_folder_path):
            os.mkdir(logs_folder_path)

        self.path_to_folder = logs_folder_path + '\\'

        episode_info_path = self.path_to_folder + Parameters.episode_data_file_name

        self.episode_info_file = open(episode_info_path, 'w+')

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

    def write_statistics(self, agent_history):

        self.write_episodic_info_using_history(agent_history)
        self.write_data_per_episode_using_history(agent_history)

    def write_episodic_info_using_history(self, agent_history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count

        for i in range(len(losses)):
           self.episode_info_file.write( self.episode_info_format 
                                         % (i, 
                                            episode_lengths[i], 
                                            losses[i], 
                                            min_rewards[i], 
                                            max_rewards[i], 
                                            rewards[i],
                                            max_cells[i], 
                                            max_cells_count[i]))
           self.episode_info_file.write('\n')

    def write_data_per_episode_using_history(self, agent_history):

        state_evolution_per_episode  = agent_history.state_evolution_per_episode
        reward_evolution_per_episode = agent_history.rewards_on_action_per_episode
        action_per_episode           = agent_history.actions_taken_per_episode
        state_evolution_per_episode  = agent_history.state_evolution_per_episode
        network_output               = agent_history.network_output
        entropy                      = agent_history.entropy_per_episode

        file_name = self.path_to_folder + 'episode_' + '%s' + '.txt'

        for i in range(len(reward_evolution_per_episode)):
            with open(file_name % i, 'w+') as episode_file:

                for j in range(len(reward_evolution_per_episode[i])):
                    episode_file.write(self.data_per_episode_format
                                    % (state_evolution_per_episode[i][j],
                                        ['%.3f' % output for output in network_output[i][j]], 
                                        entropy[i][j], 
                                        action_per_episode[i][j], 
                                        reward_evolution_per_episode[i][j]))
                    episode_file.write('\n')

    def __del__(self):
        print('Closing files')
        self.episode_info_file.close()