import torch

class NARX(torch.nn.Module):
    def __init__(self, d_i, d_o, d_x, d_y, d_hl):
        super().__init__()

        # lookback
        self.d_i = d_i
        self.d_o = d_o

        # timeseries dimensions
        self.d_x = d_x
        self.d_y = d_y

        # general parameters
        self.d_hl = d_hl

        self.hl1 = torch.nn.Linear(self.d_i * self.d_x + self.d_o * self.d_y, self.d_hl)
        self.hl2 = torch.nn.Linear(self.d_hl, self.d_y)

        self.act1 = torch.nn.Tanh()
        self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        # x shape [batch_size, num_steps, dim]        
        if x.shape[1] < self.d_i: 
            raise Exception(f"Time series length is {x.shape[1]}, while the input delay is {self.d_i}")
        
        y = torch.zeros(x.shape[0], x.shape[1]-self.d_i, self.d_y)

        for i in range(x.shape[1]-self.d_i):
            x_squashed = x[:, i:i+self.d_i, :].view(x.shape[0], self.d_i*self.d_x)
            input_stacked = torch.stack(x_squashed, y[i:i+self.d_o], dim=1)

            x_hl = self.hl1(input_stacked)
            x_hl = self.act1(x_hl)

            x_hl = self.hl2(x_hl)
            x_hl = self.act2(x_hl)

            y[i+self.d_o] = x_hl.squeeze()

        
        

        

