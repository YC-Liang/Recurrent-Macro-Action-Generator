#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Evaluation Args')
parser.add_argument('--task', required=True, help='Task')
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
from models import MAGICGen_VLMAG, MAGICCriticNet_VLMAG
from utils import Statistics, PARTICLE_SIZES, CONTEXT_SIZES

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(precision=4, suppress=True)

TASK = args.task
MAX_ACTION_LEN = 16
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] + 1 if TASK in CONTEXT_SIZES else None
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent
GEN_MODEL = args.gen_model_name

if __name__ == '__main__':
    model_path = args.model_path + '/gen_model.pt.{:08d}'
    model_path = model_path.format(args.model_index)
    open('learned_action_weights.csv', 'w').close()
    plot_heat_map = False

    print('Loading model... ({})'.format(model_path))
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if TASK in ['DriveHard']:
            gen_model = MAGICGenNet_DriveHard(MAX_ACTION_LEN, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        else:
            gen_model = MAGICGen_VLMAG(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
        if model_path is not None:
            gen_model.load_state_dict(torch.load(model_path))
        gen_model.eval()

        env = Environment(TASK, MAX_ACTION_LEN, True)

        while True:

            context = env.read_context()
            #assume the critic feedback 0 
            context = np.append(context, 0)
            print(context)
            #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]

            state = env.read_state()
            #print(state)
            if state is not None:
                macro_action_weights, macro_action_indices = gen_model.forward(
                            torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                            torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0))
                params = macro_action_indices.squeeze(0).cpu().numpy()
                #print("Weights:")
                #macro_action_weights = macro_action_weights.squeeze(0).cpu().numpy().reshape(16,73)
                #print(macro_action_weights)
                with open("learned_action_weights.csv", 'a') as log_file:
                    for row in macro_action_weights:
                        for w in row:
                            log_file.write(str(w) + ', ')
                        log_file.write('\n')
                    log_file.write('\n')
                #print("Indices:")
                #print(params)
                if plot_heat_map:
                    plt.imshow(macro_action_weights, cmap='hot', interpolation='nearest')
                    plt.show()


                env.write_params(params)
            response = env.process_response()
            if response.is_failure:
                print("Collision happened.")
            print(f"Undiscounted reward: {response.undiscounted_reward}")
