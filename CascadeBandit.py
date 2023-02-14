import time
import os
import pickle 
import datetime
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from CascadeConf import *
from datasets.genCasDataset import genCasDataset
from datasets.genMovieLensDataset import genMovieLensDataset
from BanditAlg.casUCB1 import CascadeUCB1
from BanditAlg.BatchRank import BatchRank
from BanditAlg.TopRank import TopRank
from BanditAlg.casKLUCB import CascadeKLUCB
import argparse


class simulateOnlineData:
    def __init__(self, dataset, seed_size, iterations):
        self.dataset = dataset
        self.seed_size = seed_size
        self.iterations = iterations
        
        self.startTime = datetime.datetime.now()
        # self.BatchCumlateRegret = {}
        # self.AlgRegret = {}

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        # for alg_name, alg in list(algorithms.items()):
        #     self.AlgRegret[alg_name] = []
        #     self.BatchCumlateRegret[alg_name] = []

        self.resultRecord()

        for iter_ in range(self.iterations):
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide()
                C = -1
                for i in range(len(S)):
                    arm = S[i]
                    if self.dataset.w[arm] >= random.uniform(0, 1):
                        C = i
                        break

                alg.updateParameters(C)
                # self.AlgRegret[alg_name].append(regret)
            self.resultRecord(iter_)

        self.showResult()


    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            # self.filenameWriteRegret = os.path.join(save_address, 'Regret{}.csv'.format(str(args.exp_num)))
            self.filenameWriteCost = os.path.join(save_address, 'Cost{}.csv'.format(str(args.exp_num)))
            self.filenameTargetRate = os.path.join(save_address, 'Rate{}.csv'.format(str(args.exp_num)))

            if not os.path.exists(save_address):
                os.mkdir(save_address)

            if os.path.exists(self.filenameWriteCost) or os.path.exists(self.filenameTargetRate): # or os.path.exists(self.filenameWriteRegret):
                raise ValueError ("Save File exists already, please check experiment number")

            # with open(self.filenameWriteRegret, 'w') as f:
            #     f.write('Time(Iteration)')
            #     f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
            #     f.write('\n') 

            with open(self.filenameWriteCost, 'w') as f:
                f.write('Time(Iteration)')
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(alg_name)
                f.write(',' + ','.join(l))
                f.write('\n') 

            with open(self.filenameTargetRate, 'w') as f:
                f.write('Time(Iteration)')
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(alg_name)
                f.write(',' + ','.join(l))
                f.write('\n') 
        else:
            # if run in the experiment, save the results
            print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
            self.tim_.append(iter_)
            # for alg_name in algorithms.keys():
            #     self.BatchCumlateRegret[alg_name].append(sum(self.AlgRegret[alg_name]))
            # with open(self.filenameWriteRegret, 'a+') as f:
            #     f.write(str(iter_))
            #     f.write(',' + ','.join([str(self.BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
            #     f.write('\n')

            with open(self.filenameWriteCost, 'a+') as f:
                f.write(str(iter_))
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(str(algorithms[alg_name].totalCost[-1]))
                f.write(',' + ','.join(l))
                f.write('\n')

            
            with open(self.filenameTargetRate, 'a+') as f:
                f.write(str(iter_))
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(str(algorithms[alg_name].num_targetarm_played[-1]))
                f.write(',' + ','.join(l))
                f.write('\n')

            
    def showResult(self):
        
        # # regret
        # f, axa = plt.subplots(1, sharex=True)
        # for alg_name in algorithms.keys():  
        #     axa.plot(self.tim_, self.BatchCumlateRegret[alg_name],label = alg_name)
        #     print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateRegret[alg_name])))
        # axa.legend(loc='upper left',prop={'size':9})
        # axa.set_xlabel("Iteration")
        # axa.set_ylabel("Regret")
        # axa.set_title("Average Regret")
        # plt.savefig('./SimulationResults/CascadeBandit/AvgRegret' + str(args.exp_num)+'.png')
        # plt.show()

        # plot cost
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].cost, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Cost")
        plt.savefig(save_address + '/Cost' + str(args.exp_num)+'.png')
        plt.show()

        # plot cumulative cost
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].totalCost, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Total Cost")
        plt.savefig(save_address + '/TotalCost' + str(args.exp_num)+'.png')
        plt.show()

        # plot superarm played
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].num_targetarm_played, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Count")
        axa.set_title("Number of times target arm is played")
        plt.savefig(save_address + '/TargetarmPlayed' + str(args.exp_num)+'.png')
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_num', type=int, default=0)

    args = parser.parse_args()

    if dataset == 'synthetic':
        data = genCasDataset(num_arms, seed_size)
    
    else:
        if dataset == 'movielens-small':
            data = genMovieLensDataset('./datasets/movielens/ml-latest-small/ratings.csv', seed_size, args.exp_num)

    simExperiment = simulateOnlineData(data, seed_size, iterations)

    target_arms_set = data.target_arms_set

    algorithms = {}
    # print(dataset)
    # algorithms['CascadeUCB1-Attack'] = CascadeUCB1_Attack(data, data.num_arms, seed_size, target_arms)

    algorithms['CascadeUCB1-Attack'] = CascadeUCB1(data, data.num_arms, seed_size, attack='general')
    algorithms['CascadeKLUCB-Attack'] = CascadeKLUCB(data, data.num_arms, seed_size, attack='general')
    algorithms['BatchRank-Attack'] = BatchRank(data, data.num_arms, seed_size, iterations, attack='general')
    algorithms['TopRank-Attack'] = TopRank(data, data.num_arms, seed_size, iterations, attack='general')

    algorithms['BatchRank-AttackThenQuit'] = BatchRank(data, data.num_arms, seed_size, iterations, attack='attack&quit')
    algorithms['TopRank-AttackThenQuit'] = TopRank(data, data.num_arms, seed_size, iterations, attack='attack&quit')
    
    simExperiment.runAlgorithms(algorithms)