from math import *
import numpy as np
import random
import copy


def generalAttack(arms, target_arms_set, seed_size):
	best_arms = copy.deepcopy(arms)
	cost = 0
	if len(list(set(target_arms_set).difference(set(best_arms)))) > 0:
		for i in range(seed_size):
			if best_arms[i] not in target_arms_set:
				cost = 1
				best_arms[i] = -10000

	return best_arms, cost


def AttackThenQuit(best_arms, num_arms, target_arm, seed_size, clicks):
	# best_arms = arms.tolist()
	cost = 1

	if type(clicks).__name__ == 'list':
		if target_arm in best_arms:
			C = [target_arm]

			# if best_arms.index(target_arm) not in clicks:
			# 	cost = len(clicks) + 1
			# 	C = [target_arm]
			# else:
			# 	C = [target_arm]
			# 	cost = len(clicks) - 1
		else:
			# cost = len(clicks)
			C = []

	else:
		raise ValueError("Cascade ban not impl")
	
	return C, cost