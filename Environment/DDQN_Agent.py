class DDQN_Agent():
    
    def __init__(self):
        self.policy_net = DQN(16, 4).cuda()
        self.target_net = DQN(16, 4).cuda()
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.game = Environment_2048(params.board_size)
        self.replay_memory = ReplayMemory(params.memory_size)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
    
    def predict_action_t(self, torch_state):
        
        state = list(torch_state.cpu().numpy().reshape(-1, params.board_size).astype(int))
        available_actions = self.game.getAvailableMoves(state, params.board_size)
        
        predicted_values = self.policy_net(torch_state)
        
        max_value = torch.max(predicted_values[available_actions])

        output = [index for index, value in enumerate(predicted_values) 
                  if index in available_actions and 
                  value == max_value]
        
        return torch.tensor([[random.choice(output)]], 
                            dtype = torch.long, 
                            device = params.device)
    
    
    def predict_action(self, state):
        
        available_actions = self.game.getAvailableMoves(state, params.board_size)
        
        torch_state = torch.tensor(state, dtype = torch.float32, 
                                    device = params.device).flatten()

        return self.predict_action_t(torch_state)
    
    def convert_state_to_tensor(self, state):
        return torch.tensor([state], dtype = torch.float32, 
                            device = params.device).flatten()
    
    def optimize_model(self):
        if len(self.replay_memory) < params.batch_size:
            return
        transitions = self.replay_memory.sample(params.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device = params.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_batch = state_batch.view(-1, params.board_size ** 2)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        non_final_next_states = non_final_next_states.view(-1, params.board_size ** 2)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(params.batch_size, device=params.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * params.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    def show_information(self):
        
        lengths, scores, endGameSums, maxCells =                         self.game.getEpisodesData()
        plt.figure()
        plt.plot(self.running_mean(lengths, 10))
        plt.figure()
        plt.plot(self.running_mean(scores, 10))
        plt.figure()
        plt.plot(self.running_mean(endGameSums, 10))
        plt.figure()
        plt.hist(maxCells, bins=[0, 8, 16, 32, 64, 128, 256, 512, 1024])
                            
    def play_and_learn(self, episodes = 1):
        
        self.print_headers()
        
        for e in range(episodes):
            self.game.resetGame()
            previous_score = 0
            
            state = self.game.getBoard()
            while self.game.isFinished() == False:
                action = self.predict_action(state)
                
                self.game.takeAction(action.item())
                new_score = self.game.getScore()
                
                reward = new_score - previous_score
                reward = torch.tensor([reward], 
                                      dtype = torch.float32,
                                      device = params.device)
                previous_score = new_score
                
                if self.game.isFinished() == True:
                    next_state = None
                else:
                    next_state = self.game.getBoard()
                
                torch_state = self.convert_state_to_tensor(state)
                torch_next_state = None if next_state == None else self.convert_state_to_tensor(next_state)
                self.replay_memory.push(torch_state, action, torch_next_state, reward)
                
                # Add transition - DONE
                # Perform update - DONE
                # Copy weights - DONE
                
                #CHECK IF TARGET NET TRIES TO SAMPLE SOME OTHER VALUE(NON-PERMITTED ONES) 
                state = next_state
                
                self.optimize_model()
            
            if e % params.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.game.calculateEndGameData()
            self.print_episode_info(e)
        
    def print_episode_info(self, e):
        game_length, game_score, game_egs, game_max_cell = self.game.getCurrentEpisodeData()
            
        print('=' * 120)
        print('= {:^20} = {:^20} = {:^20} = {:^20} = {:^20} = '.format(e, game_score, game_length, game_egs, game_max_cell))
    
    def print_headers(self):
        print('= {:^20} = {:^20} = {:^20} = {:^20} = {:^20} = '.format('Episode', 'Score', 'Length', 
                                                                       'End game sum', 'Max cell'))