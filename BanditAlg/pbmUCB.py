from math import *
import  numpy as np
import random

class PbmUCB():
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
		
	def decide(self):
		for i in range(self.num_arms):
			if self.N[i] == 0:
				self.U[i] = float('inf')
			else:
				self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(self.t)/self.N_tilde[i])
				self.U[i] = min(1, max(self.U[i], 0))

		self.best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]
		
		return self.best_arms
	
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
