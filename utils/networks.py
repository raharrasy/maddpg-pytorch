import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 n_type="policy"):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        self.n_type = n_type
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        if n_type == "policy":
            self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
            self.fc4 = nn.Linear(hidden_dim[2], hidden_dim[3])
            self.fc5 = nn.Linear(hidden_dim[3], hidden_dim[4])
            self.fc6 = nn.Linear(hidden_dim[4], out_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim[1], out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc6.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        if self.n_type == "policy":
            h1 = self.fc2(self.nonlin(self.fc1(self.in_fn(X))))
            h2 = self.nonlin(self.fc4(self.fc3(h1)))
            out = self.fc6(self.nonlin(self.fc5(h2)))
        else:
            h1 = self.nonlin(self.fc1(self.in_fn(X)))
            h2 = self.nonlin(self.fc2(h1))
            out = self.fc3(h2)
        return out