class Logger():
    def __init__(self, logs_path, episode_info_file, general_info_file):
        episode_info_path = logs_path + '\\' + episode_info_file
        general_info_path = logs_path + '\\' + general_info_file

        self.episode_info_file = open(episode_info_path, 'w+')
        self.general_info_file = open(general_info_path, 'w+')


    def write_information(self, agent_history):
        losses  = agent_history.get_losses()
        rewards = agent_history.get_episode_rewards()
        episode_lengths = agent_history.get_episode_lengths()

        for i in range(len(losses)):
           self.episode_info_file.write('| Episode %5s | Lengths: %-5s | Loss: %-5.2f | Reward: %-5s|' % (i, episode_lengths[i], losses[i], rewards[i]))
           self.episode_info_file.write('\n')

    def __del__(self):
        print('Closing files')
        self.episode_info_file.close()
        self.general_info_file.close()