from base64 import b64encode, b64decode
from collections import namedtuple
from subprocess import Popen, PIPE
import cv2 as cv
import numpy as np
import struct
import time
import os

Response = namedtuple('Response', [
    'best_value',
    'steps',
    'discounted_reward',
    'undiscounted_reward',
    'is_terminal',
    'is_failure',
    'macro_length',
    'num_nodes',
    'depth',
    'stats'
])

NUM_PARTICLES = 100

class Environment:

    def __init__(self, task, macro_length, visualize):
        self.task = task
        l = ['../../cpp/build/DespotMagicEnv{}'.format(task)]
        l.append('--macro-length={}'.format(macro_length))
        if visualize:
            l.append('--visualize')
        self.process = Popen(l, shell=False, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        self.context_arr = np.asarray([])
        self.state_arr = np.asarray([])

    def read_context(self):
        data = np.array([x[0] for x in struct.iter_unpack('f', b64decode(self.process.stdout.readline().decode('utf8').strip()))])
        data.astype(np.float32)
        self.context_arr = data
        return data

    def read_state(self):
        raw = self.process.stdout.readline().decode('utf8').strip()
        if raw == '':
            self.state_arr = np.asarray([])
            return None
        else:
            data = np.array([x[0] for x in struct.iter_unpack('f', b64decode(raw))]).astype(np.float32)
            self.state_arr = data
        return data

    def construct_visual(self):
        SCENARIO_MIN = -35.0
        SCENARIO_MAX = 30.0
        RESOLUTION = 0.1

        #helper function that converts world frame to opencv frame

        to_frame = lambda vec : (
            int((vec[0]-SCENARIO_MIN)/RESOLUTION),
            int((SCENARIO_MAX-vec[1])/RESOLUTION)
        )
        
        to_frame_dist = lambda d : int(d/RESOLUTION)

        x_to_frame = lambda x : int((x-SCENARIO_MIN)/RESOLUTION)
        y_to_frame = lambda y : int((SCENARIO_MAX-y)/RESOLUTION)

        #build white background
        img = np.zeros((
            int(abs(SCENARIO_MAX * 2)/RESOLUTION),
            int(abs(SCENARIO_MIN * 2)/RESOLUTION),
            3
        ), np.uint8) + 255

        #draw goal 
        cv.circle(img, 
        to_frame(self.context_arr[0:2]), 
        to_frame_dist(1.5),
        (81, 218, 11),
        -1)

        #draw lights
        lights = self.context_arr[2:16].reshape(7,2)
        for l in lights:
            cv.circle(img, 
            to_frame(l), 
            to_frame_dist(2.0),
            (26, 118, 245),
            -1)
        
        #draw wall
        walls = self.context_arr[16: 36].reshape(5,4)
        blue = np.asarray([255,0,0], np.uint8)
        for w in walls:
            cv.rectangle(img, 
            to_frame(w[0:2]),
            to_frame(w[2:]),
            tuple(blue.tolist()),
            -1)
        
        #draw danger zone
        dgr_zone = self.context_arr[36: 52].reshape(4,4)
        for d in dgr_zone:
            cv.rectangle(img,
            to_frame(d[0:2]),
            to_frame(d[2:]),
            (69, 69, 186),
            -1)
        #draw belief particles, 100 of them, the first one in an particle is the step number
        belief_particles = self.state_arr.reshape(100,3)
        for b in belief_particles:
            cv.drawMarker(img,
            to_frame(b[1:]),
            (0,255,255),
            8,
            2,
            4)
        print(belief_particles[87])
        cv.imshow('visual', img)
        #raise Exception("Not Implemented yet")
        cv.waitKey()


        



        

    def write_params(self, params):
        params_raw = b64encode(b''.join(struct.pack('f', p) for p in params))
        self.process.stdin.write((params_raw.decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()

    def process_response(self):
        best_value = float(self.process.stdout.readline().decode('utf8').strip())
        if best_value > 9999999:
            best_value = None
        steps = int(self.process.stdout.readline().decode('utf8').strip())
        discounted_reward = float(self.process.stdout.readline().decode('utf8').strip())
        undiscounted_reward = float(self.process.stdout.readline().decode('utf8').strip())
        is_terminal = float(self.process.stdout.readline().decode('utf8').strip()) > 0.5
        is_failure = float(self.process.stdout.readline().decode('utf8').strip()) > 0.5
        macro_length = int(self.process.stdout.readline().decode('utf8').strip())
        num_nodes = int(self.process.stdout.readline().decode('utf8').strip())
        if num_nodes > 9999999:
            num_nodes = None
        depth = int(self.process.stdout.readline().decode('utf8').strip())
        if depth > 9999999:
            depth = None

        stats_str = [self.process.stdout.readline().decode('utf8').strip() for _ in range(5)]
        stats = []
        for s in stats_str:
            if s == '':
                stats.append(None)
            else:
                stats.append(np.array([x[0] for x in struct.iter_unpack('f', b64decode(s))]).astype(np.float32))

        return Response(best_value, steps, discounted_reward, undiscounted_reward,
                is_terminal, is_failure, macro_length, num_nodes, depth, stats)

