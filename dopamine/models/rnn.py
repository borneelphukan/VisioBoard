from torch import nn
import torch

torch.manual_seed(0)


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(
            input_size, hidden_dim, batch_first=True, bias=True, nonlinearity="relu"
        )
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, hidden):
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        # Get last time step of the output
        r_out = r_out[:, -1, :]
        y1 = self.fc(r_out)
        return y1

