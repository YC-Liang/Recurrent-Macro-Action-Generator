import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt

import os
import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Training Plots')
parser.add_argument('--folders', required=True, default=['LightDark'])
parser.add_argument('--name', required=True, default=None)
parser.add_argument('--smooth', required=False, default=False)
args = parser.parse_args()

SMOOTH_INTERVAL = 10000
SKIP_INTERVAL = 1000
COLORS = ['blue', 'red', 'green']

def plot_training_data(folders, name):
	data_dirs = []
	for f in folders:
		data_dirs.append(f + '/' + name)

	fig = plt.figure()
	plt.xlable('Iterations')
	plt.ylable('Training Rewards')

	for d in data_dirs:
		data = genfromtxt(data_dir, delimiter = ',')
		cumulative_rewards = data[:, 1]
		rewards_with_gap = [cumulative_rewards[i] for i in range(0, data.shape[0], 1000)]
		x_axis = np.arange(rewards_with_gap.shape[0])
		x_axis = x_axis * SKIP_INTERVAL
		plt.plot(x_axis, rewards_with_gap)
	
	fig.tight_layout()
	plt.show()

	
def plot_training_data_single(folder, name, smooth = False):
	data_dir = folder + '/' + name
	data = genfromtxt(data_dir, delimiter = ',')
	raw_name = name.split('.')[0]
	#print(data.shape)

	cumulative_rewards = data[:, 1]
	rewards_with_gap = [cumulative_rewards[i] for i in range(0, data.shape[0], 1000)]
	print(len(rewards_with_gap))
	if smooth:
		smoothed_rewards = []
		for i in range(0, data.shape[0], SMOOTH_INTERVAL):
			interval = cumulative_rewards[i:i+SMOOTH_INTERVAL]
			smoothed_rewards.append(np.mean(interval))
		smoothed_rewards = np.asarray(smoothed_rewards)
		x_axis = np.arange(smoothed_rewards.shape[0])
		x_axis = x_axis * SMOOTH_INTERVAL

		plt.figure()
		plt.plot(x_axis, smoothed_rewards)
		plt.xlabel('Iterations')
		plt.ylabel('Cumulative rewards')
		plt.savefig(folder + '/' + raw_name + '_plot.png')
		plt.show()

	else:
		x_axis = np.arange(data.shape[0])
		x_axis = x_axis * 100

		plt.figure()
		plt.plot(x_axis, rewards_with_gap)
		plt.xlabel('Iterations')
		plt.ylabel('Cumulative rewards')
		plt.savefig(folder + '/' + raw_name + '_plot.png')
		plt.show()
	return

def plot_retrain_folder(folder, name):
	i = 0
	rewards = []
	raw_name = name.split('.')[0]
	while True:
		data_dir = folder + "/" + str(i) + "_" + name
		if not os.path.exists(data_dir):
			break
		data = genfromtxt(data_dir, delimiter = ',')
		rewards.append(data[:, 1])
		i+=1
	rewards = np.asarray(rewards)
	mean_rewards = np.mean(rewards, axis=0)
	std_rewards = np.std(rewards, axis=0)
	smoothed_rewards = []
	smoothed_std = []
	for i in range(0, mean_rewards.shape[0], SMOOTH_INTERVAL):
		interval = mean_rewards[i:i+SMOOTH_INTERVAL]
		std_interval = std_rewards[i:i+SMOOTH_INTERVAL]
		smoothed_std.append(np.mean(std_interval))
		smoothed_rewards.append(np.mean(interval))
	smoothed_rewards = np.asarray(smoothed_rewards)
	smoothed_std = np.asarray(smoothed_std)
	x_axis = np.arange(smoothed_rewards.shape[0])
	x_axis = x_axis * SMOOTH_INTERVAL

	plt.figure()
	plt.plot(x_axis, smoothed_rewards, 'k-')
	plt.fill_between(x_axis, smoothed_rewards-smoothed_std, smoothed_rewards+smoothed_std)
	plt.xlabel('Iterations')
	plt.ylabel('Cumulative rewards')
	plt.savefig(folder + '/' + raw_name + '_plot.png')
	plt.show()
	return
	

	



if __name__ == '__main__':
	folders = args.folders
	name = args.name 
	smooth = args.smooth
	plot_retrain_folder(folders, name)
	#plot_training_data_single(folders, name, smooth)

	#if len(folders) == 1:
	#	print('plotting single plots')
	#	plot_training_data_single(folders[0], name, smooth)
	#else:
	#	plot_training_data(folders, name)