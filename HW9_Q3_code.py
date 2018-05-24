# William Dahl
# 001273655
# ICSI431 Data mining
# May 2nd, 2018

import numpy as np
import types
import itertools
import copy
import json

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

#lables
label = {0: "Seattle", 1: "Boston", 2: "Washington D.C.", 3: "Philapedia", 4: "New York City"}
y = [[0 for i in range(10)] for j in range(1500)] #quanitfied data
i = 0 #index counter
# loops through the file 
for line in open("HW9_Q3_training.txt", "r").readlines():
    cities = json.loads(line) #loads the line into a array
    j = 0 # index counter
    # loops through each city in cities
    for city in cities:
        # checks for the city and lables the data accordingly
        if city == "Seattle":
            y[i][j] = 0

        if city == "Boston":
            y[i][j] = 1

        if city == "Washington D.C.":
            y[i][j] = 2

        if city == "Philapedia":
            y[i][j] = 3

        if city == "New York City":
            y[i][j] = 4

        j += 1

    i += 1

# train a markov model
mm = markovmodel()
mm.fit(y)

f = open ("HW9_Q3_predictions.txt", "w")# writes the predictions ot this file
#loops through the test file
for line in open("HW9_Q3_testing.txt", "r").readlines():
    test = [0 for i in range(10)] #creates the lest arrays
    cities = json.loads(line) #the current line as an array
    j = 0
    #loops through the cities
    for city in cities:
        #marks the cities
        if city == "Seattle":
            test[j] = 0

        if city == "Boston":
            test[j] = 1

        if city == "Washington D.C.":
            test[j] = 2

        if city == "Philapedia":
            test[j] = 3

        if city == "New York City":
            test[j] = 4

        j += 1

    pred = mm.predict(test, 5) #predicts
    pred_lables = [0 for i in range(5)] #holds the lables
    i = 0
    #gets the lables form the data
    for s in pred:
        pred_lables[i] = label[s]
        i += 1

    json_string = json.dumps(pred_lables)#tunrs array to string
    f.write(json_string)#writes to file
    f.write("\n")