import time
import os
import pickle 
import datetime
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PBMConf import *
from datasets.genPBMdataset import genPBMdataset
from datasets.genMovieLensDataset import genMovieLensDataset
from BanditAlg.pbmUCB import PbmUCB
from BanditAlg.pbmUCB_Attack import PbmUCB_Attack
from BanditAlg.casUCB1 import CascadeUCB1
from BanditAlg.casUCB1_Attack import CascadeUCB1_Attack
from BanditAlg.BatchRank import BatchRank
from BanditAlg.BatchRankAttack import BatchRankAttack
from BanditAlg.TopRank import TopRank
import argparse


class simulateOnlineData:
	def __init__(self, dataset, seed_size, iterations):
		self.dataset = dataset
		self.seed_size = seed_size
		self.iterations = iterations
		
		self.startTime = datetime.datetime.now()
		# self.BatchCumlateRegret = {}
		# self.AlgRegret = {}

	def runAlgorithms(self, algorithms):
		self.tim_ = []
		# for alg_name, alg in list(algorithms.items()):
			# self.AlgRegret[alg_name] = []
			# self.BatchCumlateRegret[alg_name] = []

		self.resultRecord()

		for iter_ in range(self.iterations):
			
			for alg_name, alg in list(algorithms.items()): 
				S = alg.decide()
				# regret = alg.click_prob - self.dataset.click_prob
				# print("C", regret)

				w = sorted(self.dataset.total_prob, key=lambda x: self.dataset.total_prob[x], reverse=True)
				# print("W", w)
				# print("S", S)
				# print(self.dataset.target_arms)

				click = []
				for i in range(len(S)):
					arm = S[i]
					total_click_prob = self.dataset.total_prob[arm]
					if total_click_prob >= random.uniform(0, 1):
						click.append(i)
						# break

				alg.updateParameters(S, click)

				# self.AlgRegret[alg_name].append(regret)

			self.resultRecord(iter_)

		self.showResult()


	def resultRecord(self, iter_=None):
		# if initialize
		if iter_ is None:
			# self.filenameWriteRegret = os.path.join(save_address, 'Regret{}.csv'.format(str(args.exp_num)))
			self.filenameWriteCost = os.path.join(save_address, 'Cost{}.csv'.format(str(args.exp_num)))
			self.filenameTargetRate = os.path.join(save_address, 'Rate{}.csv'.format(str(args.exp_num)))

			if not os.path.exists(save_address):
				os.mkdir(save_address)

			if os.path.exists(self.filenameWriteCost) or os.path.exists(self.filenameTargetRate):
				raise ValueError ("Save File exists already, please check experiment number")

			# with open(self.filenameWriteRegret, 'w') as f:
			# 	f.write('Time(Iteration)')
			# 	f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
			# 	f.write('\n') 

			with open(self.filenameWriteCost, 'w') as f:
				f.write('Time(Iteration)')
				l = []
				for alg_name in algorithms.keys():
					if 'Attack' in alg_name:
						l.append(alg_name)
				f.write(',' + ','.join(l))
				f.write('\n') 

			with open(self.filenameTargetRate, 'w') as f:
				f.write('Time(Iteration)')
				l = []
				for alg_name in algorithms.keys():
					if 'Attack' in alg_name:
						l.append(alg_name)
				f.write(',' + ','.join(l))
				f.write('\n') 
		else:
			# if run in the experiment, save the results
			print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
			self.tim_.append(iter_)
			# for alg_name in algorithms.keys():
			# 	# self.BatchCumlateRegret[alg_name].append(sum(self.AlgRegret[alg_name]))
			# with open(self.filenameWriteRegret, 'a+') as f:
			# 	f.write(str(iter_))
			# 	f.write(',' + ','.join([str(self.BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
			# 	f.write('\n')

			with open(self.filenameWriteCost, 'a+') as f:
				f.write(str(iter_))
				l = []
				for alg_name in algorithms.keys():
					if 'Attack' in alg_name:
						l.append(str(algorithms[alg_name].totalCost[-1]))
				f.write(',' + ','.join(l))
				f.write('\n')

			
			with open(self.filenameTargetRate, 'a+') as f:
				f.write(str(iter_))
				l = []
				for alg_name in algorithms.keys():
					if 'Attack' in alg_name:
						l.append(str(algorithms[alg_name].num_targetarm_played[-1]))
				f.write(',' + ','.join(l))
				f.write('\n')

			
	def showResult(self):
		
		# # regret
		# f, axa = plt.subplots(1, sharex=True)
		# for alg_name in algorithms.keys():  
		# 	axa.plot(self.tim_, self.BatchCumlateRegret[alg_name],label = alg_name)
		# 	print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateRegret[alg_name])))
		# axa.legend(loc='upper left',prop={'size':9})
		# axa.set_xlabel("Iteration")
		# axa.set_ylabel("Regret")
		# axa.set_title("Average Regret")
		# plt.savefig('./SimulationResults/PBMBandit/AvgRegret' + str(args.exp_num)+'.png')
		# plt.show()

		# plot cost
		f, axa = plt.subplots(1, sharex=True)
		for alg_name in algorithms.keys():
			if "Attack" in alg_name:
				axa.plot(self.tim_, algorithms[alg_name].cost, label = alg_name)
		axa.legend(loc='upper left',prop={'size':9})
		axa.set_xlabel("Iteration")
		axa.set_ylabel("Cost")
		axa.set_title("Cost")
		plt.savefig('./SimulationResults/PBMBandit/Cost' + str(args.exp_num)+'.png')
		plt.show()

		# plot cumulative cost
		f, axa = plt.subplots(1, sharex=True)
		for alg_name in algorithms.keys():
			if "Attack" in alg_name:
				axa.plot(self.tim_, algorithms[alg_name].totalCost, label = alg_name)
		axa.legend(loc='upper left',prop={'size':9})
		axa.set_xlabel("Iteration")
		axa.set_ylabel("Cost")
		axa.set_title("Total Cost")
		plt.savefig('./SimulationResults/PBMBandit/TotalCost' + str(args.exp_num)+'.png')
		plt.show()

		# plot superarm played
		f, axa = plt.subplots(1, sharex=True)
		for alg_name in algorithms.keys():
			if "Attack" in alg_name:
				axa.plot(self.tim_, algorithms[alg_name].num_targetarm_played, label = alg_name)
		axa.legend(loc='upper left',prop={'size':9})
		axa.set_xlabel("Iteration")
		axa.set_ylabel("Count")
		axa.set_title("Number of times target arm is played")
		plt.savefig('./SimulationResults/PBMBandit/TargetarmPlayed' + str(args.exp_num)+'.png')
		plt.show()



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('exp_num', type=int, default=0)

	args = parser.parse_args()

	random.seed(args.exp_num)

	if dataset == 'synthetic':
		data = genPBMdataset(num_arms, seed_size)
	
	else:
		if dataset == 'movielens-small':
			data = genMovieLensDataset('./datasets/movielens/ml-latest-small/ratings.csv', seed_size, args.exp_num)

	simExperiment = simulateOnlineData(data, seed_size, iterations)

	target_arms = data.target_arms[:-1]

	# target_arms = random.sample(range(data.num_arms), seed_size)

	algorithms = {}
	# print(dataset)
	# algorithms['BatchRank'] = BatchRank(data, data.num_arms, seed_size, target_arms, iterations)
	# algorithms['BatchRankAttack'] = BatchRankAttack(data, data.num_arms, seed_size, target_arms, iterations)

	# algorithms['CascadeUCB1'] = CascadeUCB1(data, data.num_arms, seed_size, target_arms)
	# algorithms['CascadeUCB1Attack'] = CascadeUCB1_Attack(data, data.num_arms, seed_size, target_arms)

	# algorithms['pbmUCB'] = PbmUCB(data, data.num_arms, seed_size, target_arms)
	algorithms['pbmUCBAttack'] = PbmUCB_Attack(data, data.num_arms, seed_size, target_arms)

	# algorithms['TopRank_Attack'] = TopRank(data, data.num_arms, seed_size, target_arms, iterations)
	

	simExperiment.runAlgorithms(algorithms)