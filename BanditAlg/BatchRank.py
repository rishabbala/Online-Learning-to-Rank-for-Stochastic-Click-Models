from math import *
import numpy as np
import random


class BatchRank():

	def __init__(self, dataset, num_arms, seed_size, target_arms, iterations):
		self.dataset = dataset
		self.seed_size = seed_size
		self.iterations = iterations
		self.num_arms = num_arms

		self.c = np.zeros((2*self.seed_size, self.iterations, self.num_arms))
		self.n = np.zeros((2*self.seed_size, self.iterations, self.num_arms))

		self.A = set([0])
		self.b_max = 0
		self.I = {0: [0, self.seed_size]}
		self.B = {(0, 0): range(0, self.num_arms)}
		self.l = {0: 0}
		self.best_arms = np.zeros(self.seed_size)

		self.t = 1


	def decide(self):

		np.ndarray.fill(self.best_arms, -1)

		for b in self.A:
			self.DisplayBatch(b)


		self.click_prob = 1

		for i in self.best_arms:
			self.click_prob *= (1-self.dataset.total_prob[i])

		return self.best_arms


	def DisplayBatch(self, b):

		l = self.l[b]
		n_min = float('inf')
		d_list = {}

		for d in self.B[(b, l)]:
			d_list[d] = self.n[b][l][d]
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		d_list = list(dict(sorted(d_list.items(), key=lambda x: x[1])).keys())
		policy = list(range(0, self.I[b][1]-self.I[b][0]))
		random.shuffle(policy)

		# print("D", d_list)
		# print("P", policy)

		for k in range(self.I[b][0], self.I[b][1]):
			# print("k", k, k-self.I[b][0])
			self.best_arms[k] = d_list[policy[k-self.I[b][0]]]

	
	def collectClicks(self, b, click):
		l = self.l[b]
		n_min = float('inf')

		for d in self.B[(b, l)]:
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		for k in range(self.I[b][0], self.I[b][1]):
			if self.n[b][l][int(self.best_arms[k])] == n_min:
				if k in click:
					self.c[b][l][int(self.best_arms[k])] += 1
				self.n[b][l][int(self.best_arms[k])] += 1

		# print(self.n[b][l])


	def UpperBound(self, b, d, nl):

		kl_div = lambda p, q: p*np.log(1e-10 + p/(1e-10+q)) + (1-p)*np.log(1e-10 + (1-p)/(1-q+1e-10))

		l = self.l[b]
		c_prob = self.c[b,l,d] / nl

		l_bnd = c_prob
		u_bnd = 1

		while (u_bnd-l_bnd) > 10**-3:
			if nl * kl_div(c_prob, (u_bnd+l_bnd)/2) > np.log(self.iterations) + 3 * np.log(np.log(self.iterations)):
				u_bnd = (u_bnd+l_bnd)/2
			else:
				l_bnd = (u_bnd+l_bnd)/2

		return nl * kl_div(c_prob, u_bnd)


	def LowerBound(self, b, d, nl):

		kl_div = lambda p, q: p*np.log(1e-10 + p/(1e-10+q)) + (1-p)*np.log(1e-10 + (1-p)/(1-q+1e-10))

		l = self.l[b]
		c_prob = self.c[b,l,d] / nl

		l_bnd = 0
		u_bnd = c_prob

		while (u_bnd-l_bnd) > 10**-3:
			if nl * kl_div(c_prob, (u_bnd+l_bnd)/2) > np.log(self.iterations) + 3 * np.log(np.log(self.iterations)):
				u_bnd = (u_bnd+l_bnd)/2
			else:
				l_bnd = (u_bnd+l_bnd)/2

		return nl * kl_div(c_prob, l_bnd)


	def UpdateBatch(self, b):
		l = self.l[b]
		n_min = float('inf')

		for d in self.B[(b, l)]:
			if self.n[b][l][d] < n_min:
				n_min = self.n[b][l][d]

		nl = ceil(16 * (2**(-l))**(-2) * np.log(self.iterations))

		U = {}
		L = {}

		if n_min == nl:
			for d in self.B[(b, l)]:
				U[(b, l, d)] = self.UpperBound(b, d, nl)
				L[(b, l, d)] = self.LowerBound(b, d, nl)

			d_list = {}

			for d in self.B[(b, l)]:
				d_list[d] = L[(b, l, d)]

			d_list = dict(sorted(d_list.items(), key=lambda x: x[1], reverse=True))
			d_list = list(d_list.keys())
			
			BK_plus = []
			BK_minus = []

			for k in range(1, self.I[b][1]-self.I[b][0]+1):
				BK_plus.append(d_list[:k])
				BK_minus.append(list(set(self.B[(b, l)]).difference(set(BK_plus[-1]))))

				if len(BK_minus[-1]) == 0:
					return

			# print("BK", BK_plus, BK_minus)
			# exit()

			s = 0
			for k in range(0, self.I[b][1]-self.I[b][0]):
				# print(int(d_list[k]))
				if L[(b, l, int(d_list[k]))] > max([U[(b, l, d)] for d in BK_minus[k]]):
					s = k

			# print("D", d_list, s)
			# exit()

			if s==0 and len(list(self.B[(b, l)])) > self.I[b][1]-self.I[b][0]:
				new_B = [d for d in self.B[(b, l)] if U[(b, l, d)] >= L[(b, l, d_list[self.I[b][1]-self.I[b][0]-1])]]
				if len(new_B) > 0:
					self.B[(b, l+1)] = new_B
					self.l[b] += 1
			
			elif s > 0:
				# self.A = self.A.union(set([self.b_max+1, self.b_max+1]).difference(set([b])))
				self.A = list(set(self.A).union(set([self.b_max+1, self.b_max+2])).difference(set([b])))
				self.I[self.b_max+1] = [self.I[b][0], self.I[b][0]+s+1]
				self.B[(self.b_max+1, 0)] = list(BK_plus[s])
				self.l[self.b_max+1] = 0

				self.I[self.b_max+2] = [self.I[b][0]+s+1, self.I[b][1]]
				self.B[(self.b_max+2, 0)] = list(BK_minus[s])
				self.l[self.b_max+2] = 0

				self.b_max += 2

			# print(self.A)
			# print(self.I)
			# print(self.B)
			# exit()


	def updateParameters(self, best_arms, click):
		
		for b in self.A:
			self.collectClicks(b, click)
		
		for b in self.A:
			self.UpdateBatch(b)