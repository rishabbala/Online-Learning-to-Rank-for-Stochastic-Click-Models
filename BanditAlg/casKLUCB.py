from math import *
import numpy as np
import random
from BanditAlg.attack import generalAttack


class CascadeKLUCB():

	def __init__(self, dataset, num_arms, seed_size, attack=True):
		self.dataset = dataset
		self.T = {}
		self.w_hat = {}
		self.U = {}
		self.t = 1
		self.num_arms = num_arms
		self.seed_size = seed_size
		self.attack_bool = attack

		for i in range(self.num_arms):
			self.T[i] = 0
			self.w_hat[i] = 0

		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []


	def decide(self):

		d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
		f = lambda t: t*((np.log(t))**3)

		for i in range(self.num_arms):
			l = self.w_hat[i]
			u = 1

			if self.T[i] == 0:
				self.U[i] = float('inf')
			else:
				while (u-l) > 10**-3:
					if d(self.w_hat[i], (u+l)/2) > np.log(f(self.t))/self.T[i]:
						u = (u+l)/2
					else:
						l = (u+l)/2

				self.U[i] = u
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

		return best_arms

	
	def updateParameters(self, C):

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