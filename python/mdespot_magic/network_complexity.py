#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Training Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--not-belief-dependent', dest='belief_dependent', default=True, action='store_false')
parser.add_argument('--not-context-dependent', dest='context_dependent', default=True, action='store_false')
parser.add_argument('--gen-model-name', required = False, default = "Vanilla")
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet, MAGICGenNet_DriveHard, MAGICCriticNet_DriveHard
from models import MAGICGen_Autoencoder, MAGICGen_RNN, MAGICGen_Encoder
from models import MAGICGenNet_DriveHard_RNN, MAGICGenNet_DriveHard_Encoder
from replay import ReplayBuffer
from utils import PARTICLE_SIZES, CONTEXT_SIZES

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

np.set_printoptions(precision=4, suppress=True)
torch.set_num_threads(1)

TASK = args.task
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] if TASK in CONTEXT_SIZES else None
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent

GEN_MODEL = args.gen_model_name
GEN_MODELS = ['Vanilla', 'Autoencoder', 'RNN', 'Encoder']

if GEN_MODEL not in GEN_MODELS:
    raise Exception("Invalid generative model type")

if __name__ == '__main__':
    # Load models.
    print('Loading models...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if TASK in ['DriveHard']:
        #TODO: Change model according to the command line input
        if GEN_MODEL == 'RNN':
            gen_model = MAGICGenNet_DriveHard_RNN(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
            critic_model = MAGICCriticNet_DriveHard(MACRO_LENGTH, True, True).float().to(device)
        elif GEN_MODEL == "Encoder":
            gen_model = MAGICGenNet_DriveHard_Encoder(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
            critic_model = MAGICCriticNet_DriveHard(MACRO_LENGTH, True, True).float().to(device)
        else:
            gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
            critic_model = MAGICCriticNet_DriveHard(MACRO_LENGTH, True, True).float().to(device)
    else:
        if GEN_MODEL == 'Vanilla':
            gen_model = MAGICGenNet(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
        elif GEN_MODEL == 'Autoencoder':
            gen_model = MAGICGen_Autoencoder(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
        elif GEN_MODEL == 'Encoder':
            gen_model = MAGICGen_Encoder(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
        elif GEN_MODEL == 'RNN':
            gen_model = MAGICGen_RNN(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
        else:
            raise Exception("Invalid generative model type")
    summary(gen_model, input_size=((1,3), (1,300)))
    
            

