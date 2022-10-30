#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Training Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
parser.add_argument('--num-iterations', type=int, default=None)
parser.add_argument('--not-belief-dependent', dest='belief_dependent', default=True, action='store_false')
parser.add_argument('--not-context-dependent', dest='context_dependent', default=True, action='store_false')
parser.add_argument('--output-dir', required=False, default=None)
#parser.add_argument('--gen-model-name', required = False, default = "Vanilla")
parser.add_argument('--log-dir', required=False, default=None)
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet, MAGICGenNet_DriveHard, MAGICCriticNet_DriveHard
from models import MAGICGen_VLMAG, MAGICCriticNet_VLMAG
from replay import ReplayBuffer
from utils import PARTICLE_SIZES, CONTEXT_SIZES

import cv2
import itertools
import multiprocessing
import numpy as np
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zmq
np.set_printoptions(precision=4, suppress=True)
torch.set_num_threads(1)

TASK = args.task
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK]+1 if TASK in CONTEXT_SIZES else None
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent

print(f"The model depends on context: {CONTEXT_DEPENDENT}")
print(f"The model depends on belief: {BELIEF_DEPENDENT}")

# Training configurations
REPLAY_MIN = 10000
REPLAY_MAX = 100000
REPLAY_SAMPLE_SIZE = 256
SAVE_INTERVAL = 5000 if TASK == 'DriveHard' else 10000
PRINT_INTERVAL = 100
RECENT_HISTORY_LENGTH = 50
OUTPUT_DIR = args.output_dir
LOG_DIR = args.log_dir
SAVE_PATH = 'learned_{}/'.format(TASK)
LOG_SAVE_PATH = 'learned_{}.csv'.format(TASK)
#MACRO_ACTION_LOG_PATH = '{}_macro_actions_dist.csv'.format(TASK)
NUM_ITERATIONS = args.num_iterations
NUM_CURVES = 8  #number of actions in a macro action
MAX_ACTION_LEN = 16
VOCABULARY_SIZE = 73

if TASK in ['DriveHard']:
    TARGET_ENTROPY = np.array([-1.0] * 14, dtype=np.float32)
    LOG_ALPHA_INIT = [0.0] * 14
    LR = 1e-4
elif TASK in ['PuckPush']:
    TARGET_ENTROPY = [-5.0] * NUM_CURVES
    LOG_ALPHA_INIT = [-1.0] * NUM_CURVES
    LR = 1e-4
elif TASK in ['LightDark']:
    TARGET_ENTROPY = [-5.0] * NUM_CURVES
    LOG_ALPHA_INIT = [-3.0] * NUM_CURVES
    LR = 1e-4
elif TASK in ['Navigation2D']:
    TARGET_ENTROPY = [-5.0] * NUM_CURVES
    LOG_ALPHA_INIT = [-2.0] * NUM_CURVES
    LR = 1e-4
LOG_ALPHA_MIN = -10.
LOG_ALPHA_MAX = 20.
#LAMBDA = 5

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(port):
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MAX_ACTION_LEN, False)

    steps = 0
    total_reward = 0
    previous_best_reward = -50

    while True:

        # Read from environment.
        context = environment.read_context()
        context = np.append(context, previous_best_reward)
        #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]
        state = environment.read_state()

        # Call generator if needed.
        if state is not None:
            socket.send_pyobj((
                'CALL_GENERATOR',
                context,
                state))
            macro_action_indices = socket.recv_pyobj()
            environment.write_params(macro_action_indices)

        # Read response.
        response = environment.process_response()
        previous_best_reward = response.best_value

        # Add experience.
        if state is not None and response.best_value is not None:
            socket.send_pyobj((
                'ADD_EXPERIENCE',
                context,
                state,
                macro_action_indices,
                response.best_value))
            socket.recv_pyobj()

        # Add to trajectory statistics.
        steps += response.steps
        total_reward += response.undiscounted_reward

        # Upload trajectory statistics.
        if response.is_terminal:
            collision = response.is_failure
            socket.send_pyobj((
                'ADD_TRAJECTORY_RESULT',
                steps, total_reward, collision, response.stats))
            socket.recv_pyobj()
            steps = 0
            total_reward = 0

def rand_macro_action_set(num_macros, max_length):
    weights = torch.rand(VOCABULARY_SIZE, max_length) #VOC_SIZE x MAX_ACTION_LEN
    indices = weights.topk(NUM_CURVES, 0)[1] #NUM_CURVES x MAX_ACTION_LEN
    weights = np.asarray(weights, dtype=np.float32).flatten()
    indices = np.asarray(indices, dtype=np.float32).flatten()
    return weights, indices


def distance_penalty(macro_action):
    '''
    Calculate the straighline distance covered by the macro action
    and calculate the corresponding penalty term
    '''
    zero_index = np.where(macro_action == 0)[0]
    if zero_index.shape[0] != 0:    
        macro_action = macro_action[:zero_index[0]]
    n = macro_action.shape[0]

    pos = np.array([0,0])
    for action in macro_action:
        theta= (5*action-5) * (np.pi / 180)
        pos = pos + np.array([np.cos(theta), np.sin(theta)])

    return 1 - np.linalg.norm(pos) / n


def lambda_scheduler(n):
    #return 5 * np.exp(-(n/30000))
    return 3.5


if __name__ == '__main__':

    if OUTPUT_DIR is not None:
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
            except:
                pass

    #create log file directory for saving logs
    if LOG_DIR is not None:
        if not os.path.exists(LOG_DIR):
            try:
                os.makedirs(LOG_DIR)
            except:
                pass
        #empty any previous same log file 
        open(LOG_DIR + '/' + LOG_SAVE_PATH, 'w').close()


    save_path = SAVE_PATH
    if OUTPUT_DIR is not None:
        save_path = OUTPUT_DIR + '/' + save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Load models.
    print('Loading models...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    gen_model = MAGICGen_VLMAG(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).float().to(device)
    critic_model = MAGICCriticNet_VLMAG(CONTEXT_SIZE, PARTICLE_SIZE, True, True).float().to(device)
    gen_model_optimizer = optim.Adam(gen_model.parameters(), lr=LR)
    critic_model_optimizer = optim.Adam(critic_model.parameters(), lr=LR)
    #log_alpha = torch.tensor(LOG_ALPHA_INIT, requires_grad=True, device=device)
    #alpha_optim = optim.Adam([log_alpha], lr=LR)

    # Prepare zmq server.
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    print('Starting processes...')
    processes = [multiprocessing.Process(target=environment_process, args=(port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    step = 0
    start = time.time()
    recent_steps = []
    recent_total_reward = []
    recent_collisions = []
    recent_values = []
    recent_stats = [[] for _ in range(5)]
    replay_buffer = ReplayBuffer(REPLAY_MAX)

    while True:

        # Read request and process.
        request = socket.recv_pyobj()
        instruction = request[0]
        instruction_data = request[1:]

        if instruction == 'CALL_GENERATOR':
            if len(replay_buffer) < REPLAY_MIN:
                _, macro_action_indices = rand_macro_action_set(NUM_CURVES, MAX_ACTION_LEN)
            else:
                with torch.no_grad():
                    _, macro_action_indices = gen_model.forward(
                        torch.tensor(instruction_data[0], dtype=torch.float, device=device).unsqueeze(0),
                        torch.tensor(instruction_data[1], dtype=torch.float, device=device).unsqueeze(0))
                    macro_action_indices = macro_action_indices.squeeze(0).cpu().numpy()
                    #macro_action_weights = macro_action_weights.squeeze(0).cpu().numpy()
            socket.send_pyobj(macro_action_indices)

        elif instruction == 'ADD_TRAJECTORY_RESULT':
            socket.send_pyobj(0) # Return immediately.
            recent_steps.append((instruction_data[0], instruction_data[2] > 0.5))
            recent_total_reward.append(instruction_data[1])
            recent_collisions.append(instruction_data[2])
            recent_steps = recent_steps[-RECENT_HISTORY_LENGTH:]
            recent_total_reward = recent_total_reward[-RECENT_HISTORY_LENGTH:]
            recent_collisions = recent_collisions[-RECENT_HISTORY_LENGTH:]

            for i in range(len(recent_stats)):
                if instruction_data[3][i] is not None:
                    recent_stats[i].append(instruction_data[3][i])
                    recent_stats[i] = recent_stats[i][-RECENT_HISTORY_LENGTH:]

        elif instruction == 'ADD_EXPERIENCE':
            socket.send_pyobj(0) # Return immediately.

            recent_values.append(instruction_data[3])
            recent_values = recent_values[-RECENT_HISTORY_LENGTH:]

            # Add to buffer.
            instruction_data_cuda = [torch.tensor(t, dtype=torch.float, device=device) for t in instruction_data]
            replay_buffer.append(instruction_data_cuda)

            # Check for minimum replay size.
            if len(replay_buffer) < REPLAY_MIN:
                print('Waiting for minimum buffer size ... {}/{}'.format(len(replay_buffer), REPLAY_MIN))
                continue

            # Sample training mini-batch.
            sampled_evaluations = replay_buffer.sample(REPLAY_SAMPLE_SIZE)
            sampled_contexts = torch.stack([t[0] for t in sampled_evaluations])
            sampled_states = torch.stack([t[1] for t in sampled_evaluations])
            sampled_params = torch.stack([t[2] for t in sampled_evaluations])
            sampled_values = torch.stack([t[3] for t in sampled_evaluations])

            # Update critic.
            critic_loss = torch.distributions.Normal(*critic_model(sampled_contexts, sampled_states, sampled_params)) \
                    .log_prob(sampled_values).mean(dim=-1)
            critic_model_optimizer.zero_grad()
            gen_model_optimizer.zero_grad()
            (-critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
            critic_model_optimizer.step()

            # Update params model.
            #For RNN, we require the LSTM to use new memories since DRQN reports in the long run, LSTM still learns the sequential relationship
            _, macro_action_indices = gen_model.forward(sampled_contexts, sampled_states)
            (value, sd) = critic_model(sampled_contexts, sampled_states, macro_action_indices)
            critic_model_optimizer.zero_grad()
            gen_model_optimizer.zero_grad()
            #dual_terms = (log_alpha.exp().detach() * macro_actions_entropy).sum(dim=-1)
            #calculate distance penalties for each action
            #for macro_action in macro_action_indices.squeeze(0).cpu().numpy():
            macro_action_indices = macro_action_indices.squeeze(0).cpu().numpy()
            macro_action_indices = macro_action_indices.reshape(REPLAY_SAMPLE_SIZE, NUM_CURVES, -1)
            #calculate the distance penalty term
            phi = np.zeros(REPLAY_SAMPLE_SIZE) #distance penalty terms

            for batch in range(REPLAY_SAMPLE_SIZE):
                macro_actions = macro_action_indices[batch]
                for macro_action in macro_actions:
                    phi[batch] = phi[batch] + distance_penalty(macro_action)

            phi *= lambda_scheduler(step)
            #print(phi)
            #phi_copy = phi.copy()
            phi = torch.from_numpy(phi).float().to(value.get_device())
            gen_objective = value - phi #+ dual_terms
            (-gen_objective.mean()).backward()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
            gen_model_optimizer.step()

            # Update dual variables.
            #alpha_optim.zero_grad()
            #alpha_loss = log_alpha * ((macro_actions_entropy - torch.tensor(
                    #TARGET_ENTROPY, device=device, dtype=torch.float32)).detach())
            #alpha_loss.mean().backward()
            #with torch.no_grad():
                #log_alpha.grad *= (((-log_alpha.grad >= 0) | (log_alpha >= LOG_ALPHA_MIN)) &
                        #((-log_alpha.grad < 0) | (log_alpha <= LOG_ALPHA_MAX))).float()
            #alpha_optim.step()

            # Log statistics.
            if step % PRINT_INTERVAL == 0:
                print("\033[H\033[J")
                print('Step {}: Recent Steps (Pass) = {}'.format(step,
                    np.mean([s[0] for s in recent_steps if not s[1]]) if len([s[0] for s in recent_steps if not s[1]]) > 0 else None))
                print('Step {}: Recent Steps (Fail) = {}'.format(step,
                    np.mean([s[0] for s in recent_steps if s[1]]) if len([s[0] for s in recent_steps if s[1]]) > 0 else None))
                print('Step {}: Recent Total Reward = {}'.format(step, np.mean(recent_total_reward) if len(recent_total_reward) > 0 else None))
                print('Step {}: Recent Collisions = {}'.format(step, np.mean(recent_collisions) if len(recent_collisions) > 0 else None))
                print('Step {}: Recent Values = {}'.format(step, np.mean(recent_values) if len(recent_collisions) > 0 else None))
                print('Step {}: Critic Net Loss = {}'.format(step, critic_loss.detach().item()))
                print('Step {}: Generator Mean = {}'.format(step, value.mean().detach().item()))
                print('Step {}: Generator S.D. = {}'.format(step, sd.mean().detach().item()))
                print('Step {}: Distance penalty = {}'.format(step, -phi.mean().detach().item()))
                #print('Step {}: Generator Curve Entropy = {}'.format(step, macro_actions_entropy.mean(dim=-2).detach().cpu().numpy()))
                for i in range(5):
                    chained = list(itertools.chain.from_iterable(recent_stats[i]))
                    print('Step {}: Recent Stat{} = {}'.format(step, i, np.mean(chained) if len(chained) > 0 else None))
                print('Step {}: Elapsed = {} m'.format(step, (time.time() - start) / 60))
                #print('Alpha = ', torch.exp(log_alpha))

            # save logs as csv file
            if LOG_DIR != None:
                with open(LOG_DIR + '/' + LOG_SAVE_PATH, 'a') as log_file:
                    #step, reward, collision, value, GPU memory, Critic Net Loss, Macro Action Mean, Macro Action SD, Curve Entropy
                    log_file.write(str(step) + ',')
                    log_file.write(str(np.mean(recent_total_reward)) if len(recent_total_reward) > 0 else 'NULL')
                    log_file.write(',')
                    log_file.write(str(np.mean(recent_collisions)) if len(recent_collisions) > 0 else 'NULL')
                    log_file.write(',')
                    log_file.write(str(np.mean(recent_values)) if len(recent_values) > 0 else 'NULL')
                    log_file.write(',')
                    if torch.cuda.device_count() > 0:
                        log_file.write(str(torch.cuda.memory_allocated(0)))
                        log_file.write(',')
                    log_file.write(str(critic_loss.detach().item()))
                    log_file.write(',')
                    log_file.write(str(-phi.mean().detach().item()))
                    log_file.write('\n')

            # Save models.
            if step % SAVE_INTERVAL == 0:
                print('Saving models....')
                torch.save(gen_model.state_dict(), save_path + 'gen_model.pt.{:08d}'.format(step))
                torch.save(critic_model.state_dict(), save_path + 'critic_model.pt.{:08d}'.format(step))

            if NUM_ITERATIONS is not None and step >= NUM_ITERATIONS:
                for p in processes:
                    p.terminate()
                    p.join()
                socket.close()
                exit()

            step += 1