from math import *
import numpy as np
import random
import copy
from BanditAlg.attack import generalAttack, AttackThenQuit


class Partition:
	def __init__(self, items, k, m):
		self.items = items
		self.k = k
		self.m = m

class TopRank():

	def __init__(self, dataset, num_arms, seed_size, iterations, attack):
		self.dataset = dataset
		self.num_arms = num_arms
		self.seed_size = seed_size
		self.iterations = iterations
		self.attack_type = attack

		self.S = np.zeros((self.num_arms, self.num_arms)) # S[i,j] = \sum_t U(t,i,j)
		self.N = np.zeros((self.num_arms, self.num_arms)) # N[i,j] = \sum_t |U(t,i,j)| = \sum_t |C(t,i) - C(t,j)| 1{i,j in the same partition}

		self.G = np.zeros((self.num_arms, self.num_arms), dtype = bool) # G[i,j] = 1 iff i is better than j
		self.partitions = {0:Partition(items=range(self.num_arms), k=0, m=self.seed_size)}

		self.t = 1

		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []



	def decide(self):
		self.best_arms = [-1 for _ in range(self.seed_size)]
		for c in self.partitions:
			partition = self.partitions[c]
			# print(partition.k, partition.m)
			partition.items = np.random.permutation(partition.items)
			self.best_arms[partition.k:partition.k+partition.m] = partition.items[:partition.m]
		
		best_arms = copy.deepcopy(self.best_arms)
		cost = 0

		if self.attack_type == 'general':
			best_arms, cost = generalAttack(self.best_arms, self.dataset.target_arms_set, self.seed_size)

			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)

		# print(self.best_arms, best_arms, self.dataset.target_arms_set, self.dataset.target_arm)

		return best_arms


	def _criterion(self, S, N):
		c = 3.4367
		return S >= np.sqrt(2 * N * np.log(c * np.sqrt(self.iterations) * np.sqrt(N)))


	def updateCascade(self, clicks):
		# update S and N
		for c in self.partitions:
			partition = self.partitions[c]
			for i in range(partition.m):
				a = partition.items[i]
				for j in range(i+1, len(partition.items)):
					b = partition.items[j]
					if (a in self.best_arms and self.best_arms.index(a) <= clicks) or (b in self.best_arms and self.best_arms.index(b) <= clicks):
						ca = cb = 0
						if a in self.best_arms and clicks == self.best_arms.index(a):
							ca = 1
						if b in self.best_arms and clicks == self.best_arms.index(b):
							cb = 1
						x = ca - cb
						self.S[a, b] += x
						self.N[a, b] += np.abs(x)
						self.S[b, a] -= x
						self.N[b, a] += np.abs(x)

	def updatePBM(self, clicks):
		# update S and N
		for c in self.partitions:
			partition = self.partitions[c]
			for i in range(partition.m):
				a = partition.items[i]
				for j in range(i+1, len(partition.items)):
					b = partition.items[j]
					ca = cb = 0
					if a in clicks:
						ca = 1
					if b in clicks:
						cb = 1
					x = ca - cb
					self.S[a, b] += x
					self.N[a, b] += np.abs(x)
					self.S[b, a] -= x
					self.N[b, a] += np.abs(x)


	def updateParameters(self, clicks):

		cost = 0
		c = 3.43
		delta = 1/self.iterations
		T1 = np.ceil(4 * np.log(c/delta)/ (self.seed_size/self.num_arms + (1-np.sqrt(1+8*self.seed_size/self.num_arms))/4))


		if self.attack_type == 'attack&quit' and self.t <= T1:
			clicks, cost = AttackThenQuit(self.best_arms, self.num_arms, self.dataset.target_arm, self.seed_size, clicks)
		
		if len(self.cost) < self.t:
			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)


		if type(clicks).__name__ == 'list':
			## PBMBandit
			self.updatePBM(clicks)
		else:
			self.updateCascade(clicks)
		
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

		self.numTargetPlayed()
		self.t += 1

	
	def numTargetPlayed(self):
		n = 0
		if self.best_arms[0] == self.dataset.target_arm:
			n = 1

		if len(self.num_targetarm_played) == 0:
			self.num_targetarm_played.append(n)
		else:
			self.num_targetarm_played.append(self.num_targetarm_played[-1] + n)
