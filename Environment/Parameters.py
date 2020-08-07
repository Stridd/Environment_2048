class Parameters:
    
    def __init__(self):
        self.epsi = 0.1
        self.gamma = 0.9
        self.lr = 0.01
        self.batch_size = 128
        self.epochs = 200
        self.target_update = 10
        self.board_size = 4
        self.memory_size = 10000
        self.device = 'cuda'
