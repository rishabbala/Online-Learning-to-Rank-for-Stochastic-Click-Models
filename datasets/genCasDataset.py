import random


class genCasDataset():

    def __init__(self, num_arms, seed_size):
        self.w = {}
        self.num_arms = num_arms
        for i in range(num_arms):
            self.w[i] = random.uniform(0, 1)

            # if i <= seed_size:
            #     self.w[i] = random.uniform(2/(3*seed_size), 1/(seed_size))
            # else:
            #     self.w[i] = random.uniform(0, 1/(3*seed_size))

        # # print(self.w)
        # # exit()

        self.best_arms = sorted(self.w, key=lambda x: self.w[x], reverse=True)[:seed_size]
        self.target_arms = sorted(self.w, key=lambda x: self.w[x])[:seed_size]

        self.click_prob = 1
        for i in self.best_arms:
            self.click_prob *= (1 - self.w[i])