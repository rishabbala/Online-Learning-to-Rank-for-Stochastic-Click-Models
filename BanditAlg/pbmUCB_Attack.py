from math import *
import  numpy as np
import random
import copy

class PbmUCB_Attack():
	def __init__(self, dataset, num_arms, seed_size, target_arms):
		self.dataset = dataset
		self.num_arms = num_arms
		self.seed_size = seed_size
		self.target_arms = target_arms

		self.t = 1
		self.S = {}
		self.N_tilde = {}
		self.N = {}
		self.U = {}

		self.beta = [1/(k+1) for k in range(self.num_arms)]

		for i in range(self.num_arms):
			self.S[i] = 0
			self.N[i] = 0
			self.N_tilde[i] = 0
		
		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []
		
	def decide(self):
		for i in range(self.num_arms):
			# self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(self.t)/self.N_tilde[i])
			if self.N[i] == 0:
				self.U[i] = float('inf')
			else:
				self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(self.t)/self.N_tilde[i])
			# 	self.U[i] = min(1, max(self.U[i], 0))

		

		self.best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]
		# s = list()
		# for k in self.best_arms:
		# 	s.append(self.U[k])
		# print("SBU",s)


		best_arms = copy.deepcopy(self.best_arms)

		cost = 0
		if len(list(set(self.dataset.target_arms).difference(set(self.best_arms)))) > 0:
			for i in range(self.seed_size):
				if self.best_arms[i] not in self.dataset.target_arms:
					cost += 1
					best_arms[i] = -10000
					# self.click_prob *= (1-self.dataset.total_prob[best_arms[i]])

		# for i in self.best_arms:
		# 	self.click_prob *= (1-self.dataset.total_prob[i])

		if len(self.totalCost) == 0:
			self.totalCost = [cost]
		else:
			self.totalCost.append(self.totalCost[-1] + cost)
		self.cost.append(cost)

		return best_arms
	
	def updateParameters(self, C,best_arms):
		if type(C).__name__ == 'list':
			for i in range(self.seed_size):
				arm = self.best_arms[i]
				self.N[arm] += 1
				self.N_tilde[arm] += self.beta[i]
				if arm in C:
					self.S[arm] += 1
		else:
			for i in range(self.seed_size):
				arm = self.best_arms[i]
				self.N[arm] += 1
				self.N_tilde[arm] += self.beta[i]
		
		self.t += 1

		self.numTargetPlayed()

	
	def numTargetPlayed(self):
		num_basearm_played = 0
		num_targetarm_played = 0

		# s = list()
		# print("SB", self.best_arms)
		# print("ST", self.dataset.target_arms)
		# for k in self.dataset.target_arms:
		# 	s.append(self.U[k])
		# print("STU",s)

		# if self.cost[-1] == 0 and self.best_arms[0] == self.dataset.target:
		# 	num_targetarm_played += 1

		if self.best_arms[0] == self.dataset.target:
			num_targetarm_played += 1

		# print("B", num_targetarm_played)
		
		if len(self.num_targetarm_played) == 0:
			self.num_targetarm_played.append(num_targetarm_played)
		else:
			self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)
