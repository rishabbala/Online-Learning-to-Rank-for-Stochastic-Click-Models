import random


class genPBMDataset():

    def __init__(self, num_arms, seed_size):
        self.w = {-10000: 0}

        self.num_arms = num_arms
        for i in range(num_arms):
            self.w[i] = random.uniform(0.25, 1)

        self.best_arms = sorted(self.w, key=lambda x: self.w[x], reverse=True)[:seed_size]
        self.target_arm = sorted(self.w, key=lambda x: self.w[x], reverse=True)[5]
        self.target_arms_set = [self.target_arm] + sorted(self.w, key=lambda x: self.w[x], reverse=True)[10:10+seed_size-1]
        
        self.examination_prob = [1/(i+1) for i in range(seed_size)]

        self.click_prob = 1
        for i in self.best_arms:
            self.click_prob *= (1 - self.w[i])