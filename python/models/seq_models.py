#!/usr/bin/env python3

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from power_spherical import PowerSpherical
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

MACRO_CURVE_ORDER = 3
NUM_CURVES = 8
EPS = 0.01
VOCABULARY_SIZE = 361 #discretise angles into 1 degree each. The first represents the <EOS> token
MAX_ACTION_LEN= 24

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def LeakyHardTanh(x, x_min=-1, x_max=1, hard_slope=1e-2):
    return (x >= x_min) * (x <= x_max) * x + (x < x_min) * (x_min + hard_slope * (x - x_min)) + (x > x_max) * (x_max + hard_slope * (x - x_max))

def LeakyReLUTop(x, x_max=1, hard_slope=1e-2):
    return (x <= x_max) * x + (x > x_max) * (x_max + hard_slope * (x - x_max))

#Defining the Variable Length Macro Action Generator
class MAGICGen_VLMAG(nn.Module):
    def __init__(self, context_size, particle_size, context_dependent, belief_dependent):
        print("Using the Sequential RNN...")
        super(MAGICGen_VLMAG, self).__init__()
        self.context_size = context_size
        self.particle_size = particle_size
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.context_dependent or self.belief_dependent:

            if self.belief_dependent:
                self.fc1 = nn.Linear(particle_size, 256)
                self.fc2 = nn.Linear(256, 256)

            self.fc3 = nn.Linear(context_size * self.context_dependent + 256 * self.belief_dependent, 512)
            self.fc4 = nn.Linear(512, 256)
            self.fc5 = nn.Linear(256, 128)
            self.fc6 = nn.Linear(128, 32)
            self.lstm = nn.LSTM(32, 32, 6)

            # Output layer.
            self.fc10_mean = nn.Linear(32, VOCABULARY_SIZE)
            torch.nn.init.constant_(self.fc10_concentration.bias, 100)

        else:
            self.mean = nn.Parameter(torch.normal(torch.zeros(NUM_CURVES * (2 * MACRO_CURVE_ORDER)), 1), requires_grad=True)
            self.concentration = nn.Parameter(100 * torch.ones(NUM_CURVES), requires_grad=True)

    def forward(self, c, x, hidden, cell):

        #TODO: ensure dimension works, check particle size, batch size etc.

        x = x.reshape((x.shape[0], -1, self.particle_size))


        # FC for each particle.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=-2)

        x = torch.cat(
                ([c] if self.context_dependent else []) \
                + ([x] if self.belief_dependent else []), dim=-1)

        # Mean latent variables -> latent dstribution parameters.
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        predictions = []
        for _ in range(MAX_ACTION_LEN):   
            #print(f'Input shape to LSTM {x.size()}')
            if hidden.numel() == 0 or cell.numel() == 0:
                #print("LSTM using new memory")
                x, (hidden, cell) = self.lstm(x)
            else:
                #print("LSTM using previous memory")
                x, (hidden, cell) = self.lstm(x, (hidden, cell))
            #print(f'Output shape to LSTM {x.size()}')
            mean = self.fc10_mean(x)
            target = mean.topk(NUM_CURVES, 1)[1]
            predictions.append(target)
        predictions.transpose(0, 1)

        return predictions.view((-1, MAX_ACTION_LEN * NUM_CURVES)) #returns a single list 


    def rsample(self, c, x, hidden, cell):
        (macro_actions_dist,), hidden, cell = self.forward(c, x, hidden, cell)
        macro_actions = macro_actions_dist.rsample()
        macro_actions = macro_actions.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))
        macro_actions_entropy = macro_actions_dist.entropy()

        return (macro_actions, macro_actions_entropy), hidden, cell

    def mode(self, c, x, hidden, cell):
        (macro_actions_dist,), hidden, cell = self.forward(c, x, hidden, cell)
        macro_actions = macro_actions_dist.loc.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))

        return macro_actions, hidden, cell


class MAGICCriticNet_VLMAG(nn.Module):
    def __init__(self, context_size, particle_size, context_dependent, belief_dependent):
        super(MAGICCriticNet_VLMAG, self).__init__()
        self.context_size = context_size
        self.particle_size = particle_size
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.belief_dependent:
            self.fc1 = nn.Linear(particle_size, 256)
            self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(
                context_size * self.context_dependent \
                + 256 * self.belief_dependent \
                + NUM_CURVES * MAX_ACTION_LEN, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 2)
        self.fc10.bias.data[1] = 100

    def forward(self, c, x, action):

        x = x.reshape((x.shape[0], -1, self.particle_size))

        if self.belief_dependent:
            # FC for each particle.
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = x.mean(dim=-2)

        x = torch.cat(
                ([c] if self.context_dependent else []) \
                + ([x] if self.belief_dependent else []) \
                + [action], dim=-1)

        x_skip = x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_skip = x = F.relu(self.fc5(x) + x_skip)
        x = F.relu(self.fc6(x))
        x_skip = x = F.relu(self.fc7(x) + x_skip)
        x = F.relu(self.fc8(x))
        x_skip = x = F.relu(self.fc9(x) + x_skip)
        x = self.fc10(x)

        return (x[...,0], EPS + F.softplus(x[...,1]))




# Transformer code below
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MagicGen_Transformer(nn.Module):
    def __init__(self, sequence_len, context_size, particle_size, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        '''
        Inputs are passed as individual trajectories. An trajectory contains past state particles and contexts.
        A pre-processing channel is used to process each particle in each time step, then these partciles from 
        an array that is then concatenated with the context. These arrays are then passed to a transformer to 
        generate macro actions.
        '''
        super().__init__()
        self.context_size = context_size
        self.particle_size = particle_size
        self.sequence_len = sequence_len
        dim += self.context_size

        self.fc1 = nn.Linear(particle_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_len+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_mean = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, NUM_CURVES * 2 * MACRO_CURVE_ORDER)
        )

        self.mlp_concentration = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, NUM_CURVES)
        )

    def forward(self, c, x):
        x = x.reshape((x.shape[0], x.shape[1], -1, self.particle_size)) #256 4 100 3
        B, H, W, P = x.shape #Batch size, sequence len, number of particles, particle size 
        assert H == self.sequence_len, f"Invalid sequence length, expected {self.sequence_len}, got {H}"
        assert P == self.particle_size, f"Invalid particle size, expected {self.particle_size}, got {P}"
        assert c.shape[1] == self.context_size, f"Invalid context size, expected {self.context_size}, got {c.shape[1]}"

        sequence = torch.tensor([]).float().to(device)
        for i in range(H):
            hist = x[:, i, :, :]
            hist = F.relu(self.fc1(hist))
            hist = F.relu(self.fc2(hist))
            hist = hist.mean(dim=-2)
            hist = torch.cat((hist, c), dim=-1)
            hist = hist.unsqueeze(1)
            if len(sequence) == 0:
                sequence = hist
            else:
                sequence = torch.cat((sequence, hist), 1)
            #sequence = torch.cat((sequence, hist), 1) #Batch size x Sequence length x Number of preprocessed particles

        b, n, _ = sequence.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        sequence = torch.cat((cls_tokens, sequence), dim=1)
        sequence += self.pos_embedding[:, :(n + 1)]
        sequence = self.dropout(sequence)

        x = self.transformer(sequence)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        mean = self.mlp_mean(x)
        mean = mean.view((-1, NUM_CURVES, 2 * MACRO_CURVE_ORDER))
        mean = mean / (mean**2).sum(dim=-1, keepdim=True).sqrt()
        concentration = 1 + F.softplus(self.mlp_concentration(x))
        
        return (PowerSpherical(mean, concentration),)

    def rsample(self, c, x):
        (macro_actions_dist,) = self.forward(c, x)
        macro_actions = macro_actions_dist.rsample()
        macro_actions = macro_actions.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))
        macro_actions_entropy = macro_actions_dist.entropy()

        return (macro_actions, macro_actions_entropy)

    def mode(self, c, x):
        (macro_actions_dist,) = self.forward(c, x)
        macro_actions = macro_actions_dist.loc.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))

        return macro_actions

            

