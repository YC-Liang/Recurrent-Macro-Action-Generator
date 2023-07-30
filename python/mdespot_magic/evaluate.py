#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Evaluation Args')
parser.add_argument('--task', required=True, help='Task')
parser.add_argument('--macro-length', type=int, required=True, help='Macro-action length')
parser.add_argument('--model-path', required=False, help='Path to model file')
parser.add_argument('--model-index', type=int, default=None, help='Index of model to benchmark. Leave empty to benchmark all in folder.')
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
from utils import Statistics, PARTICLE_SIZES, CONTEXT_SIZES

import torch
import numpy as np
import cv2
np.set_printoptions(precision=4, suppress=True)

TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] if TASK in CONTEXT_SIZES else None
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent
GEN_MODEL = args.gen_model_name

if __name__ == '__main__':
    model_path = args.model_path + '/gen_model.pt.{:08d}'
    model_path = model_path.format(args.model_index)

    print('Loading model... ({})'.format(model_path))
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using device... {device}")
        if TASK in ['DriveHard']:
            gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        else:
            if GEN_MODEL == 'Vanilla':
                gen_model = MAGICGenNet(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
            elif GEN_MODEL == 'Encoder':
                gen_model = MAGICGen_Encoder(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
            elif GEN_MODEL == 'RNN':
                gen_model = MAGICGen_RNN(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        print(f"model path: {model_path}")
        if model_path is not None:
            print("loading the checkpoint...")
            gen_model.load_state_dict(torch.load(model_path))
        gen_model.eval()

        print("Initialise environemnt...")
        env = Environment(TASK, MACRO_LENGTH, True)
        hidden_state = torch.tensor([], device=device)
        cell_state = torch.tensor([], device=device)

        while True:

            context = env.read_context()
            #print(context)
            #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]

            state = env.read_state()
            if state is not None:
                if GEN_MODEL in ['RNN']:
                    (macro_actions), hidden_state, cell_state = gen_model.mode(
                            torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                            torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0),
                            hidden_state, cell_state)
                else:
                    (macro_actions) = gen_model.mode(
                            torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                            torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0),
                            )
                params = macro_actions.squeeze(0).cpu().numpy()
                print(params)
                env.write_params(params)
            response = env.process_response()
            if response.is_terminal:
                hidden_state = torch.tensor([], device=device)
                cell_state = torch.tensor([], device=device)
            if response.is_failure:
                print("Collision happened.")
            print(f"Undiscounted reward: {response.undiscounted_reward}")
