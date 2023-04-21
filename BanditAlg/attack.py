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
				cost += 1
				best_arms[i] = -10000

	return best_arms, cost

# used to reduce the non target arm click
def generalAttack_2(best_arms, target_arm, clicks, t):
	# best_arms = arms.tolist()
	C = copy.deepcopy(clicks)
	cost = 0
	if t <= 2000:
		if type(clicks).__name__ == 'list':
			for i in range(len(clicks)):
				if clicks[i] != target_arm:
					C.remove(clicks[i]) # remove non target arm click
					cost = 1

		else:
			if target_arm in best_arms:
				if clicks != best_arms.index(target_arm):
					if clicks == -1:
						cost = 1
					else:
						cost = 2
				C = -1 # if non target arm got clicked, change first click to -1(non arm get pulled)
					
			else:
				if clicks != -1:
					cost = 1
				C = -1 # if non target arm got clicked, change first click to -1(non arm get pulled)

	return C, cost

# used to increase the click of target arm
def generalAttack_3(best_arms, target_arm, clicks, t):
	
	C = copy.deepcopy(clicks)
	cost = 0
	if t <= 2000:
		if type(clicks).__name__ == 'list':
			if target_arm in best_arms:
				if target_arm not in clicks:
					cost = len(clicks) + 1
					C.append(target_arm) # if target arm in R_t but not clciked, get it clicked


		else:
			if target_arm in best_arms:
				if clicks != best_arms.index(target_arm):
					if clicks == -1:
						cost = 1
					else:
						cost = 2
				C = best_arms.index(target_arm) # if target arm in R_t but not clciked, get it clicked

	return C, cost

# def generalAttack_2(best_arms, target_arms_set, seed_size, clicks):
# 	# print('B:',best_arms)
# 	# print('TAS:',target_arms_set)
# 	# print('C:',clicks)
# 	# best_arms = arms.tolist()
# 	cost = 0
# 	if type(clicks).__name__ == 'list':
# 		C = copy.deepcopy(clicks)
# 		if len(list(set(target_arms_set).difference(set(best_arms)))) > 0:
# 			for i in range(seed_size):
# 				if best_arms[i] not in target_arms_set and (best_arms[i] in clicks):
# 					cost += 1
# 					C.remove(best_arms[i])

# 	else:
# 		C = copy.deepcopy(clicks)
# 		if len(list(set(target_arms_set).difference(set(best_arms)))) > 0:
# 			if len(list(set(target_arms_set).difference(set(best_arms)))) == seed_size:
# 				if C != -1:
# 					cost = 1
# 					C = -1
# 			else:
# 				if best_arms[clicks] not in target_arms_set:
# 					cost = 1
# 					C = -1
# 	# print('CC:',C)
	
# 	return C, cost


def AttackThenQuit(best_arms, num_arms, target_arm, seed_size, clicks):
	# best_arms = arms.tolist()

	if type(clicks).__name__ == 'list':
		cost = 0
		if target_arm in best_arms:
			if target_arm not in clicks:
				cost = len(clicks) + 1
				C = [target_arm]
			else:
				C = [target_arm]
				cost = len(clicks) - 1
		else:
			cost = len(clicks)
			C = []

	else:
		cost = 0
		if target_arm in best_arms:
			if clicks != best_arms.index(target_arm):
				if clicks == -1:
					cost = 1
				else:
					cost = 2
			C = best_arms.index(target_arm)
				
		else:
			if clicks != -1:
				cost = 1
			else:
				cost = 0
			C = -1

	return C, cost