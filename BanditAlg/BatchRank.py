from math import *
import numpy as np
import random
import copy
from BanditAlg.attack import generalAttack, AttackThenQuit



class BatchRank():

	def __init__(self, dataset, num_arms, seed_size, iterations, attack):
		self.dataset = dataset
		self.seed_size = seed_size
		self.iterations = iterations
		self.num_arms = num_arms
		self.attack_type = attack

		self.c = np.zeros((2*self.seed_size, min(1000, self.iterations), self.num_arms))
		self.n = np.zeros((2*self.seed_size, min(1000, self.iterations), self.num_arms))

		self.A = set([0])
		self.b_max = 0
		self.I = {0: [0, self.seed_size]}
		self.B = {(0, 0): range(0, self.num_arms)}
		self.l = {0: 0}

		self.t = 1

		self.totalCost = []
		self.cost = []
		self.num_targetarm_played = []


	def decide(self):

		self.best_arms = [-1 for _ in range(self.seed_size)]

		for b in self.A:
			self.DisplayBatch(b)

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


	def DisplayBatch(self, b):

		l = self.l[b]
		n_min = float('inf')
		d_list = {}

		for d in self.B[(b, l)]:
			d_list[d] = self.n[b][l][d]
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		d_list = list(dict(sorted(d_list.items(), key=lambda x: x[1])).keys())
		policy = list(range(self.I[b][1]-self.I[b][0]))
		random.shuffle(policy)

		for k in range(self.I[b][0], self.I[b][1]):
			self.best_arms[k] = d_list[policy[k-self.I[b][0]]]

	
	def collectClicksCascade(self, b, clicks):
		l = self.l[b]
		n_min = float('inf')

		for d in self.B[(b, l)]:
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		for k in range(self.I[b][0], self.I[b][1]):
			if k > clicks:
				break
			if self.n[b][l][int(self.best_arms[k])] == n_min:
				if k == clicks:
					self.c[b][l][int(self.best_arms[k])] += 1
				self.n[b][l][int(self.best_arms[k])] += 1



	def collectClicksPBM(self, b, clicks):
		l = self.l[b]
		n_min = float('inf')

		for d in self.B[(b, l)]:
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		for k in range(self.I[b][0], self.I[b][1]):
			if self.n[b][l][int(self.best_arms[k])] == n_min:
				if int(self.best_arms[k]) in clicks:
					self.c[b][l][int(self.best_arms[k])] += 1
				self.n[b][l][int(self.best_arms[k])] += 1


	def KL_Div(self, p, q):
		if q == 0 or q == 1:
			if p == q:
				return 0
			else:
				return float('inf')

		if p == 0:
			return (1-p)*np.log((1-p)/(1-q))

		if p == 1:
			return p*np.log(p/q)

		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))



	def UpperBound(self, b, d, nl):

		l = self.l[b]
		c_prob = self.c[b,l,d] / nl

		l_bnd = c_prob
		u_bnd = 1

		# print("BBnda", l_bnd, u_bnd, self.c[b,l,d], nl, l)

		while (u_bnd-l_bnd) > 10**-3:
			if nl * self.KL_Div(c_prob, (u_bnd+l_bnd)/2) > np.log(self.iterations) + 0 * np.log(np.log(self.iterations)):
				u_bnd = (u_bnd+l_bnd)/2
			else:
				l_bnd = (u_bnd+l_bnd)/2

			# print("B", u_bnd, l_bnd)

		return l_bnd


	def LowerBound(self, b, d, nl):

		l = self.l[b]
		c_prob = self.c[b,l,d] / nl

		l_bnd = 0
		u_bnd = c_prob

		while (u_bnd-l_bnd) > 10**-3:
			if nl * self.KL_Div(c_prob, (u_bnd+l_bnd)/2) > np.log(self.iterations) + 0 * np.log(np.log(self.iterations)):
				l_bnd = (u_bnd+l_bnd)/2
			else:
				u_bnd = (u_bnd+l_bnd)/2

		return u_bnd


	def UpdateBatch(self, b):
		l = self.l[b]
		n_min = float('inf')

		for d in self.B[(b, l)]:
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		nl = ceil(16 * (2**(-l))**(-2) * np.log(self.iterations))

		U = {}
		L = {}

		# print("N", n_min, nl)

		if n_min >= nl:
			for d in self.B[(b, l)]:
				# print("B", self.UpperBound(b, d, nl), self.LowerBound(b, d, nl))
				U[(b, l, d)] = self.UpperBound(b, d, n_min)
				L[(b, l, d)] = self.LowerBound(b, d, n_min)
				# print("UL", U[(b, l, d)], L[(b, l, d)], l)

			d_list = {}

			for d in self.B[(b, l)]:
				d_list[d] = L[(b, l, d)]

			d_list = dict(sorted(d_list.items(), key=lambda x: x[1], reverse=True))
			d_list = list(d_list.keys())
			
			BK_plus = {}
			BK_minus = {}

			for k in range(1, self.I[b][1]-self.I[b][0]):
				BK_plus[k] = d_list[:k]
				BK_minus[k] = list(set(self.B[(b, l)]).difference(set(BK_plus[k])))

			s = 0
			for k in range(1, self.I[b][1]-self.I[b][0]):
				# print("L", L[(b, l, int(d_list[k-1]))], max([U[(b, l, d)] for d in BK_minus[k]]))
				if L[(b, l, int(d_list[k-1]))] > max([U[(b, l, d)] for d in BK_minus[k]]):
					s = k

			# print("S", s, self.B[(b, l)], self.I[b][1]-self.I[b][0])

			if s==0 and len(list(self.B[(b, l)])) > self.I[b][1]-self.I[b][0]:
				# print("B", self.B[b, l])
				# for d in self.B[(b, l)]:
				# 	print("U", U[(b, l, d)], L[(b, l, d_list[self.I[b][1]-self.I[b][0]-1])], d, d_list[self.I[b][1]-self.I[b][0]-1])
				new_B = [d for d in self.B[(b, l)] if U[(b, l, d)] >= L[(b, l, d_list[self.I[b][1]-self.I[b][0]-1])]]
				if len(new_B) > 0:
					self.B[(b, l+1)] = new_B
					self.l[b] += 1

					# print("LB", self.l[b])
					self.l[b] %= 1000 ## to reduce storage space, rewrite old items
					self.c[:, self.l[b], :] = 0
					self.n[:, self.l[b], :] = 0

				# print(new_B)
				# print(d_list_full)
				# exit()
			
			elif s>0:
				self.A = list(set(self.A).union(set([self.b_max+1, self.b_max+2])).difference(set([b])))
				self.I[self.b_max+1] = [self.I[b][0], self.I[b][0]+s]
				self.B[(self.b_max+1, 0)] = BK_plus[s]
				self.l[self.b_max+1] = 0

				self.I[self.b_max+2] = [self.I[b][0]+s, self.I[b][1]]
				self.B[(self.b_max+2, 0)] = BK_minus[s]
				self.l[self.b_max+2] = 0

				self.b_max += 2

		# print(self.I)
		# print(self.B)
		# # print(self.l)
		# exit()



	def updateParameters(self, clicks):

		cost = 0
		T1 = ceil(16*self.num_arms*np.log(self.iterations))

		if self.attack_type == 'attack&quit' and self.t <= T1:
			clicks, cost = AttackThenQuit(self.best_arms, self.num_arms, self.dataset.target_arm, self.seed_size, clicks)
		
		if len(self.cost) < self.t:
			if len(self.totalCost) == 0:
				self.totalCost = [cost]
			else:
				self.totalCost.append(self.totalCost[-1] + cost)
			self.cost.append(cost)


		if type(clicks).__name__ == 'list':
			## PBMBandit Model
			for b in self.A:
				self.collectClicksPBM(b, clicks)
		
		else:
			## CascadeBandit Model
			if clicks == -1:
				clicks = self.num_arms + 1
			for b in self.A:
				self.collectClicksCascade(b, clicks)

		for b in self.A:
			self.UpdateBatch(b)

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