class DQN(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        layer_size_1 = 200
        layer_size_2 = 400
        layer_size_3 = 400
        layer_size_4 = 200
        
        self.linear_1 = nn.Linear(input_size, layer_size_1)
        self.linear_2 = nn.Linear(layer_size_1, layer_size_2)
        self.linear_3 = nn.Linear(layer_size_2, layer_size_3)
        self.linear_4 = nn.Linear(layer_size_3, output_size)
        
    def forward(self, x):
        
        sample = x.clone()
        
        out_1 = self.linear_1(sample)
        out_1 = F.relu(out_1)
        
        out_2 = self.linear_2(out_1)
        out_2 = F.relu(out_2)
        
        out_3 = self.linear_3(out_2)
        out_3 = F.relu(out_3)
        
        out_4 = self.linear_4(out_3)
        return out_4   