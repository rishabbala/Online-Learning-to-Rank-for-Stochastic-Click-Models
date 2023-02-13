from math import *
import  numpy as np
import random
from BanditAlg.attack import generalAttack



class PBMUCB():
	def __init__(self, dataset, num_arms, seed_size, attack=True):
		self.dataset = dataset
		self.num_arms = num_arms
		self.seed_size = seed_size

		self.t = 1
		self.S = {}
		self.N_tilde = {}
		self.N = {}
		self.U = {}
		self.attack_bool = attack

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
			if self.N[i] == 0:
				self.U[i] = float('inf')
			else:
				self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(self.t)/self.N_tilde[i])
				self.U[i] = min(1, max(self.U[i], 0))

		self.best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]
		
		if self.attack_bool:
			best_arms, cost = generalAttack(self.best_arms, self.dataset.target_arms_set, self.seed_size)
		else:
			best_arms = copy.deepcopy(self.best_arms)
			cost = 0

		if len(self.totalCost) == 0:
			self.totalCost = [cost]
		else:
			self.totalCost.append(self.totalCost[-1] + cost)
		self.cost.append(cost)

		# print(self.best_arms, best_arms, self.dataset.target_arms_set, self.dataset.target_arm)

		return best_arms
	
	def updateParameters(self, C):
		for i in range(self.seed_size):
			arm = self.best_arms[i]
			self.N[arm] += 1
			self.N_tilde[arm] += self.beta[i]
			if arm in C:
				self.S[arm] += 1
		
		self.t += 1
		self.numTargetPlayed()

	
	def numTargetPlayed(self):
		n = 0
		if self.best_arms[0] == self.dataset.target_arm:
			n = 1

		if len(self.num_targetarm_played) == 0:
			self.num_targetarm_played.append(n)
		else:
			self.num_targetarm_played.append(self.num_targetarm_played[-1] + n)