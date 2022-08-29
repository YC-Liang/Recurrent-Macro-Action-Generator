from replay import ReplayBuffer
import numpy as np
import torch

data_size = 20 
data_dimension = 5
replay_buffer = ReplayBuffer(20)

for _ in range(data_size):
	data = torch.rand(data_dimension)
	replay_buffer.append(data)

samples = replay_buffer.sample(5)
sample_data_1 = torch.stack([t[0] for t in samples])

print(samples)
print(sample_data_1)
print(sample_data_1.size())

