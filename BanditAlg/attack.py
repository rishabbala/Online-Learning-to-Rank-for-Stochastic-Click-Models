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