# William Dahl
# 001273655
# ICSI431 Data mining
# May 2nd, 2018
# coding: utf-8

# In[2]:

import numpy as np
import types
import itertools
import copy

# In[23]:

def get_all_sequences(m, n):
    i = 1
    S = []
    for j in range(n):
        S.append([j])
    while i < m:
        S1 = []
        for s in S:
            for j in range(n):
                s1 = copy.deepcopy(s)
                s1.append(j)
                S1.append(s1)
        S.extend(S1)
        i = i + 1
    S = [item for item in S if len(item) == m]
    return S


class markovmodel:
    #transmat: None
    def __init__(self, transmat = None, startprob = None):
        self.transmat = transmat
        self.startprob = startprob
    # It assumes the state number starts from 0
    def fit(self, X):
        ns = max([max(items) for items in X]) + 1
        self.transmat  = np.zeros([ns, ns])
        self.startprob = np.zeros([ns])
        for items in X:
            n = len(items)
            self.startprob[items[0]] += 1
            for i in range(n-1):
                self.transmat[items[i], items[i+1]] += 1
        self.startprob = self.startprob / sum(self.startprob)
        n = self.transmat.shape[0]
        d = np.sum(self.transmat, axis=1)
        for i in range(n):
            if d[i] == 0:
                self.transmat[i,:] = 1.0 / n
        d[d == 0] = 1
        self.transmat = self.transmat * np.transpose(np.outer(np.ones([ns,1]), 1./d))

#    def predict_most_probable_sequene(self, ):


    def predict(self, obs, steps):
        n = len(obs)
        if len(obs) > 0:
            combs = get_all_sequences(steps, len(self.startprob))
            max_seq = []
            max_prob = -1
            for comb in combs:
                prob = 1.0
                prev = obs[-1]
                for i in comb:
                    prob = prob * self.transmat[prev, i]
                    prev = i
                if prob > max_prob:
                    max_seq = comb
                    max_prob = prob
            return max_seq
        else:
            combs = get_all_sequences(steps, len(self.startprob))
            max_seq = []
            max_prob = -1
            for comb in combs:
                prob = 1.0
                prev = -1
                for i in comb:
                    if prev == -1:
                        prob = prob * self.startprob[i]
                    else:
                        prob = prob * self.transmat[prev, i]
                    prev = i
                if prob > max_prob:
                    max_seq = comb
                    max_prob = prob
            return max_seq


# In[1]:

label = {0: "Albany", 1: "Boston", 2: "Washington D.C.", 3: "Philadelphia", 4: "New York City"}
y = [[0, 2, 0, 1, 0], 
     [0, 4, 1, 4, 0], 
     [1, 0, 3, 1, 4],
     [0, 2, 1, 0, 2],
     [0, 3, 0, 2, 4],
     [0, 4, 0, 3, 1],
     [1, 0, 4, 3, 0],
     [1, 0, 2, 3, 1],
     [4, 3, 1, 3, 2],
     [3, 4, 3, 4, 3]]

# train a markov model
mm = markovmodel()
mm.fit(y)


# predict a sequence 4 states without a current sequence
pred = mm.predict([], 4)
print [label[s] for s in pred]

#predict a sequence of 4 states with the current state being 2
pred = mm.predict([2], 4)
print [label[s] for s in pred]





