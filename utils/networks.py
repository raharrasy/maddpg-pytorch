import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.backend.pytorch import scatter_row, gather_row
import builtins

def as_scalar(data):
    return data.item()

def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, torch.Tensor):
        max_len = as_scalar(lengths.max())
    else:
        max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    device = input.device
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = torch.tensor(index).to(device)
    return scatter_row(x, index, input).view(batch_size, max_len, *old_shape[1:])


def pack_padded_tensor(input, lengths):
    batch_size, max_len = input.shape[:2]
    device = input.device
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = torch.tensor(index).to(device)
    return gather_row(input.view(batch_size * max_len, -1), index)

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True,
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

class DGNController(nn.Module):
    def __init__(self, dim_feature, act_dims, hidden_dims, num_heads=8,
                 tau=0.3):
        super(DGNController, self).__init__()
        self.mlp1a = nn.Linear(dim_feature, hidden_dims[0])
        self.mlp1b = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mlp2a = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.mlp2b = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.tau = tau

        self.mlp_last = MultiHeadAttention(hidden_dims[3], num_heads, hidden_dims[4], tau=tau)
        self.mlp_last2 = nn.Linear(hidden_dims[3] + hidden_dims[4], act_dims)

    def forward(self, input, num_nodes):
        out = F.relu(self.mlp1a(input))
        out = self.mlp1b(out)
        out = self.mlp2a(out)

        out = F.relu(self.mlp2b(out))
        DGN_out, attention_probs = self.mlp_last(out, num_nodes)
        out1 = self.mlp_last2(F.relu(torch.cat([DGN_out, out], dim=-1)))

        return out1, attention_probs

class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so on."""

    def __init__(self, d_model, num_heads, d_head, tau):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.tau = tau
        self.proj_q = nn.Linear(d_model, num_heads * d_head)
        self.proj_k = nn.Linear(d_model, num_heads * d_head)
        self.proj_v = nn.Linear(d_model, num_heads * d_head)
        self.proj_o = nn.Linear(num_heads * d_head, d_head)

    def forward(self, x, lengths_x):
        """
        Compute multi-head self-attention.
        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_x)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(x).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(x).view(-1, self.num_heads, self.d_head)

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = pad_packed_tensor(queries, lengths_x, 0)
        keys = pad_packed_tensor(keys, lengths_x, 0)
        values = pad_packed_tensor(values, lengths_x, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = self.tau * torch.einsum('bxhd,byhd->bhxy', queries, keys)
        # normalize
        # e = e / np.sqrt(self.d_head)

        # generate mask
        mask = torch.zeros(batch_size, max_len_x, max_len_mem).to(e.device)
        for i in range(batch_size):
            mask[i, :lengths_x[i], :lengths_x[i]].fill_(1)
        mask = mask.unsqueeze(1)
        e.masked_fill_(mask == 0, 0)
        e = e + 1e-12

        # apply softmax
        alpha = torch.softmax(e, dim=-1)
        # sum of value weighted by alpha
        out = torch.einsum('bhxy,byhd->bxhd', alpha, values)
        # project to output
        out = self.proj_o(
            out.contiguous().view(batch_size, max_len_x, self.num_heads * self.d_head))
        # pack tensor
        out = pack_padded_tensor(out, lengths_x)
        x = F.relu(out)

        index = []
        for i, l in enumerate(lengths_x):
            index.extend(range(i * max_len_x, i * max_len_x + l))

        returned_alpha = alpha.permute(0, 2, 3, 1)

        returned_alpha_filtered = returned_alpha.reshape(-1, returned_alpha.shape[2],returned_alpha.shape[3])[index]
        final_alpha = returned_alpha_filtered.permute(0, 2, 1)

        return x, final_alpha.reshape(-1, final_alpha.shape[2])

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, merge='cat', tau=0.3):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, hid_dim))
        self.merge = merge

        if self.merge == "cat":
            self.fc_out = nn.Linear(hid_dim * num_heads, out_dim)
        else:
            self.fc_out = nn.Linear(hid_dim, out_dim)

        self.tau = tau

    def forward(self, graph, h):
        head_outs = [attn_head(graph, h, tau=self.tau) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return F.relu(self.fc_out(torch.cat(head_outs, dim=1)))
        else:
            # merge using average
            return F.relu(self.fc_out(torch.mean(torch.stack(head_outs))))

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc_q = nn.Linear(in_dim, out_dim)
        self.fc_m = nn.Linear(in_dim, out_dim)
        self.fc_k = nn.Linear(in_dim, out_dim)

    def edge_attention(self, edges, tau=1.0):
        # edge UDF for equation (2)
        z2 = torch.sum(tau * edges.src['z_k'] * edges.dst['z_q'], dim=1, keepdim=True)
        return {'e': z2}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z_m'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h, tau=1.0):
        # equation (1)
        z_q = self.fc_q(h)
        z_k = self.fc_k(h)
        z_m = self.fc_m(h)

        graph.ndata['z_q'] = z_q
        graph.ndata['z_k'] = z_k
        graph.ndata['z_m'] = z_m

        # equation (2)
        edge_attention_func = lambda x: self.edge_attention(edges=x, tau=tau)
        graph.apply_edges(edge_attention_func)
        # equation (3) & (4)
        graph.update_all(self.message_func, self.reduce_func)
        results = graph.ndata['h']

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())

        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return results
