#!/usr/bin/env python3

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from power_spherical import PowerSpherical

MACRO_CURVE_ORDER = 3
NUM_CURVES = 8
EPS = 0.01
VOCABULARY_SIZE = 361 #discretise angles into 1 degree each. The first represents the <EOS> token
MAX_ACTION_LEN= 24

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