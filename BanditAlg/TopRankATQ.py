from math import *
import numpy as np
import random
import copy


class Partition:
	def __init__(self, items, k, m):
		self.items = items
		self.k = k
		self.m = m

class TopRankATQ():

	def __init__(self, dataset, num_arms, seed_size, target_arms, iterations):
		self.dataset = dataset
		self.num_arms = num_arms
		self.seed_size = seed_size
		self.target_arms = target_arms
		self.iterations = iterations

		self.S = np.zeros((self.num_arms, self.num_arms)) # S[i,j] = \sum_t U(t,i,j)
		self.N = np.zeros((self.num_arms, self.num_arms)) # N[i,j] = \sum_t |U(t,i,j)| = \sum_t |C(t,i) - C(t,j)| 1{i,j in the same partition}

		self.G = np.zeros((self.num_arms, self.num_arms), dtype = bool) # G[i,j] = 1 iff i is better than j
		self.partitions = {0:Partition(items=range(self.num_arms), k=0, m=self.seed_size)}

		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []

		self.t = 1

	def decide(self):
		self.best_arms = [0] * self.seed_size
		for c in self.partitions:
			partition = self.partitions[c]
			partition.items = np.random.permutation(partition.items)
			self.best_arms[partition.k:partition.k+partition.m] = partition.items[:partition.m]
		
		# best_arms = copy.deepcopy(self.best_arms)

		# cost = 0
		# if len(list(set(self.dataset.target_arms).difference(set(self.best_arms)))) > 0:
		# 	for i in range(self.seed_size):
		# 		if self.best_arms[i] not in self.dataset.target_arms:
		# 			cost += 1
		# 			best_arms[i] = -10000

		# if len(self.totalCost) == 0:
		# 	self.totalCost = [cost]
		# else:
		# 	self.totalCost.append(self.totalCost[-1] + cost)
		# self.cost.append(cost)

		return self.best_arms


	def _criterion(self, S, N):
		c = 3.43
		return S >= np.sqrt(2 * N * np.log(c * np.sqrt(self.iterations) * np.sqrt(N)))


	def updateParameters(self, C, best_arms):

		# Threshold_1 = (4*np.log(pow(self.num_arms)*self.iterations))/((self.seed_size/self.num_arms)+(1-np.sqrt(1+8*(self.seed_size/self.num_arms)/4)))
		Threshold_1 = 4000

		cost = 0
		if self.t <= Threshold_1:
			if type(C).__name__ == 'list':
				if self.dataset.target in self.best_arms:
					if self.best_arms.index(self.dataset.target) not in C:
						cost = len(C) + 1
						C = [self.best_arms.index(self.dataset.target)]
					else:
						C = [self.best_arms.index(self.dataset.target)]
						cost = len(C) - 1
				else:
					cost = len(C)
					C = []
			else:
				if self.dataset.target not in self.best_arms:
					C = -1
					cost = 1
				elif self.best_arms[C] != self.dataset.target:
					C = self.best_arms.index(self.dataset.target)
					cost = 1
		if len(self.totalCost) == 0:
				self.totalCost = [cost]
		else:
			self.totalCost.append(self.totalCost[-1] + cost)
		self.cost.append(cost)

		clicks = np.zeros(self.num_arms)
		# list C for pbm model
		if type(C).__name__ == 'list':
			if len(C) > 0:
				for i in range(len(C)):
					clicks[self.best_arms[C[i]]] = 1
		elif C != -1: 
			clicks[self.best_arms[C]] = 1
		
		# print("SB", self.best_arms)
		# print("ST", self.dataset.target_arms)
		# print("C", C)
		# print("clicks:",clicks)

		# update S and N
		for c in self.partitions:
			partition = self.partitions[c]
			for i in range(partition.m):
				a = partition.items[i]
				for j in range(i+1, len(partition.items)):
					b = partition.items[j]
					x = clicks[a] - clicks[b]
					self.S[a, b] += x
					self.N[a, b] += np.abs(x)
					self.S[b, a] -= x
					self.N[b, a] += np.abs(x)

		updateG = False
		for c in self.partitions:
			partition = self.partitions[c]
			for i in range(partition.m):
				a = partition.items[i]
				for j in range(i+1, len(partition.items)):
					b = partition.items[j]

					if self.N[a, b] > 0:
						if self._criterion(self.S[a, b], self.N[a, b]): # a is better than b
							self.G[a, b] = True
							updateG = True
						elif self._criterion(self.S[b, a], self.N[b, a]):
							self.G[b, a] = True
							updateG = True


		if updateG:
			self.partitions = {}
			c = 0
			k = 0
			remain_items = set(np.arange(self.num_arms))
			while k < self.seed_size:
				bad_items = set(np.flatnonzero(np.sum(self.G[np.asarray(list(remain_items)),:], axis=0)))
				good_items = remain_items - bad_items
				self.partitions[c] = Partition(items=list(good_items),k=k,m=min(len(good_items), self.seed_size-k))

				k += len(good_items)
				remain_items = remain_items.intersection(bad_items)
				c += 1
			# print([self.partitions[x].items for x in self.partitions])
			# exit()

		self.t += 1
		self.numTargetPlayed()

	
	def numTargetPlayed(self):
		num_basearm_played = 0
		num_targetarm_played = 0

		# print("SB", self.best_arms)
		# print("ST", self.dataset.target_arms)

		if self.best_arms[0] == self.dataset.target:
			num_targetarm_played += 1

		# print("B", num_targetarm_played)
		
		if len(self.num_targetarm_played) == 0:
			self.num_targetarm_played.append(num_targetarm_played)
		else:
			self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)