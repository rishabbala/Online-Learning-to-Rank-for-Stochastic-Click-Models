import random


class genPBMdataset():

    def __init__(self, num_arms, seed_size):
        self.w = {-10000: 0}
        self.x = {-10000: 0}

        self.total_prob = {}

        self.num_arms = num_arms
        for i in range(num_arms):
            self.w[i] = random.uniform(0.5, 1) ## click prob
            self.x[i] = random.uniform(0.5, 1) ## examination prob
            # self.x[i] = 1/(i+1+random.uniform(0, 0.01))

            self.total_prob[i] = self.w[i]*self.x[i]

        self.best_arms = sorted(self.total_prob, key=lambda x: self.total_prob[x], reverse=True)[:seed_size]
        self.target = sorted(self.total_prob, key=lambda x: self.total_prob[x])[seed_size-1]
        self.target_arms = sorted(self.total_prob, key=lambda x: self.total_prob[x])[:seed_size][::-1]

        self.click_prob = 1
        for i in self.best_arms:
            self.click_prob *= (1 - self.total_prob[i])

        self.total_prob[-10000] = 0