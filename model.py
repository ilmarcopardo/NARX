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

    def forward(self, x, y = None):
        # x shape [batch_size, num_steps, dim]        
        if x.shape[1] < self.d_i: 
            raise Exception(f"Time series length is {x.shape[1]}, while the input delay is {self.d_i}")
        
        y_pred = torch.zeros(x.shape[0], x.shape[1], self.d_y)

        for i in range(x.shape[1]-self.d_i):

            x_squashed = x[:, i:i+self.d_i, :].view(x.shape[0], self.d_i*self.d_x)

            if y is not None: y_squashed = y[:, i:i+self.d_o, :].view(y.shape[0], self.d_o*self.d_y)
            else: y_squashed = y_pred[:, i:i+self.d_o, :].view(y_pred.shape[0], self.d_o*self.d_y)
            input_cat = torch.cat((x_squashed, y_squashed), dim=1)

            x_hl = self.hl1(input_cat)
            x_hl = self.act1(x_hl)

            x_hl = self.hl2(x_hl)
            #x_hl = self.act2(x_hl)

            y_pred[:, i+self.d_o, :] = x_hl
        
        return y_pred
    
if __name__ == "__main__":
    batch_dim = 1
    x_length = 10
    d_x = 3
    d_y = 1
    x = torch.rand(batch_dim, x_length, d_x)
    model = NARX(d_i = 3, d_o = 3, d_x = d_x, d_y = d_y, d_hl = 8)

    # mode 1
    y = model(x)
    assert y.shape == (batch_dim, x_length, d_y), f"Unexpected shape: {y.shape}, expected ({batch_dim}, {x_length}, {d_y})"

    # mode 2
    y_true = torch.rand(batch_dim, x_length, d_y)
    y = model(x, y_true)
    assert y.shape == (batch_dim, x_length, d_y), f"Unexpected shape: {y.shape}, expected ({batch_dim}, {x_length}, {d_y})"

        
        

        

