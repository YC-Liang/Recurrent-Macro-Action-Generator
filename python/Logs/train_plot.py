import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Training Plots')
parser.add_argument('--folder', required=True, default='LightDark')
parser.add_argument('--name', required=True, default=None)
args = parser.parse_args()


def plot_training_data(folder, name):
	data_dir = folder + '/' + name
	data = genfromtxt(data_dir, delimiter = ',')
	raw_name = name.split('.')[0]
	#print(data.shape)

	cumulative_rewards = data[:, 1]
	x_axis = np.arange(data.shape[0])
	x_axis = x_axis * 100

	plt.figure()
	plt.plot(x_axis, cumulative_rewards)
	plt.xlabel('Iterations')
	plt.ylabel('Cumulative rewards')
	plt.savefig(folder + '/' + raw_name + '_plot.png')
	plt.show()
	return


if __name__ == '__main__':
	folder = args.folder
	name = args.name 

	if folder == None or name == None:
		raise Exception("Not a valid directory provided")

	plot_training_data(folder, name)