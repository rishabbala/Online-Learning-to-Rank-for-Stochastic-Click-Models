from math import *
import numpy as np
import random
import copy
from BanditAlg.attack import generalAttack
from BanditAlg.attack import generalAttack_2
from BanditAlg.attack import generalAttack_3


class CascadeUCB1():

	def __init__(self, dataset, num_arms, seed_size, attack='gerneral_2'):
		self.dataset = dataset
		self.T = {}
		self.w_hat = {}
		self.U = {}
		self.t = 1
		self.num_arms = num_arms
		self.seed_size = seed_size
		self.attack = attack

		for i in range(self.num_arms):
			self.T[i] = 0
			self.w_hat[i] = 0

		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []


	def decide(self):
		for i in range(self.num_arms):
			if self.T[i] == 0:
				self.U[i] = float('inf')
			else:
				self.U[i] = self.w_hat[i] + np.sqrt(1.5*np.log(self.t)/self.T[i])
				self.U[i] = min(1, max(self.U[i], 0))
		
		self.best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]

		if self.attack == 'general_1':
			best_arms, cost = generalAttack(self.best_arms, self.dataset.target_arms_set, self.seed_size)
		else:
			best_arms = copy.deepcopy(self.best_arms)
			cost = 0

		if self.attack == 'general_1':
			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)

		# print(self.best_arms, best_arms, self.dataset.target_arms_set, self.dataset.target_arm)

		return best_arms

	
	def updateParameters(self, C):

		if self.attack == 'general_2':
			C, cost = generalAttack_2(self.best_arms, self.dataset.target_arm, C, self.t)
			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)

		if self.attack == 'general_3':
			C, cost = generalAttack_3(self.best_arms, self.dataset.target_arm, C, self.t)
			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)

		if C == -1:
			C = self.num_arms

		for i in range(self.seed_size):
			if i <= C:
				arm = self.best_arms[i]
				
				r = self.T[arm]*self.w_hat[arm]
				if i == C:
					r += 1
				self.w_hat[arm] = r/(self.T[arm]+1)
				self.T[arm] += 1
			
			else:
				break

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