import random
import random
import numpy as np
import pickle
import scipy as sp
from scipy.sparse import linalg
import scipy.sparse as spp



class genMovieLensDataset():

    def __init__(self, data_dir, seed_size, exp_num, d=20):

        data = self.load_sparse_matrix(data_dir).toarray() ## make a matrix where 1 at (i, j) implies user i reviewd movie j

        train_elements = random.sample(range(data.shape[0]), int(data.shape[0]/2))
        test_elements = list(set(range(data.shape[0])) - set(train_elements))

        self.data_train = []
        self.data_test = []
        for i in train_elements:
            self.data_train.append(data[i])
        for i in test_elements:
            self.data_test.append(data[i])

        self.data_train = np.vstack(self.data_train)
        self.data_test = np.vstack(self.data_test)

        U, D, V_transpose = np.linalg.svd(self.data_train, full_matrices=False)

        if d != None:
            U = U[:, :d]
            D = D[:d]
            V_transpose = V_transpose[:d, :]

        D = np.diag(D)

        phi_matrix = V_transpose.T @ D
        
        max_norm = max(np.linalg.norm(phi_matrix, axis=1))
        phi_matrix /= max_norm

        theta = 2 * max_norm * U.T / data.shape[0]
        theta = theta @ np.ones((int(data.shape[0]/2), 1))

        means = phi_matrix @ theta

        np.random.seed(exp_num)
        random.seed(exp_num)

        means = means.reshape(-1)
        means = np.sort(means)[::-1]

        # for i in means:
        #     if i > 0.1:
        #         print(i)
        # exit()

        ## subsample best movies
        means = means[:5000]
        np.random.shuffle(means)

        self.w = {}
        self.num_arms = 0

        for i in range(means.shape[0]):
            # print(means[i])
            self.num_arms += 1
            self.w[i] = min(1, max(0, means[i] ))

        self.best_arms = list(dict(sorted(self.w.items(), key=lambda x: x[1], reverse=True)).keys())[:seed_size]

        means_05 = np.where(means>0.1)[0].tolist()
        self.target_arms = random.sample(means_05, seed_size)

        # self.target_arms = list(dict(sorted(self.w.items(), key=lambda x: x[1], reverse=True)).keys())[seed_size:2*seed_size]

        self.click_prob = 1
        for i in self.best_arms:
            self.click_prob *= (1 - self.w[i])


    def load_sparse_matrix(self, filename):
        t = 0
        f = open(filename, 'r')
        f.readline()

        user_list = []
        item_list = []
        rating_list = []

        for line in f:
            word = line.split(',')
            user_list.append(int(word[0]))
            item_list.append(int(word[1]))
            rating_list.append((int(word[0]), int(word[1]), float(word[2])))


            t += 1
            if t % 50000 == 0:
                # print('.', end = '')
                if t % 2000000 == 0:
                    print('%d m lines' %(t // 1000000))


        users = list(set(user_list))
        items = list(set(item_list))
        user2id = dict(zip(users, range(len(users))))
        item2id = dict(zip(items, range(len(items))))

        ratings = np.zeros((len(users), len(items)))

        for r in rating_list:
            u = user2id[r[0]]
            i = item2id[r[1]]
            ratings[u, i] = r[2]


        data_list = list(np.ones(t))
        user_id_list = [user2id[u] for u in user_list]
        item_id_list = [item2id[i] for i in item_list]

        result = spp.csr_matrix((data_list, (user_id_list, item_id_list)))
        return result