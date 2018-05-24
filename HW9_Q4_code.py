# William Dahl
# 001273655
# ICSI431 Data mining
# May 2nd, 2018
# coding: utf-8

# In[3]:

import numpy as np
from hmmlearn import hmm
import types
import json
import warnings
warnings.filterwarnings("ignore")


# In[4]:

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

    def predict(self, obs, steps):
        pred = []
        n = len(obs)
        if len(obs) > 0:
            s = obs[-1]
        else:
            s = np.argmax(np.random.multinomial(1, self.startprob.tolist(), size = 1))
        for i in range(steps):
            s1 = np.random.multinomial(1, self.transmat[s,:].tolist(), size = 1)
            pred.append(np.argmax(s1))
            s = np.argmax(s1)
        return pred

# In[28]:

def hmm_predict_further_states(ghmm, obs, steps):
    y = ghmm.predict(obs)
    mm = markovmodel(ghmm.transmat_, ghmm.startprob_)
    return mm.predict([y[-1]], steps)

def hmm_predict_future_features(ghmm, obs, steps):
    y = ghmm.predict(obs)
    pred = []
    mm = markovmodel(ghmm.transmat_, ghmm.startprob_)
    sts = mm.predict([], steps)
    for s in sts:
        mean = ghmm.means_[y[-1]]
        cov = ghmm.covars_[y[-1],:]
        x = np.random.multivariate_normal(mean,cov,1)
        pred.append(x[0].tolist())
    return pred

# X: sequence of observations
# y: sequence of latent states
def estimate_parameters(X, y):
    mm = markovmodel()
    mm.fit(y)
    data = dict()
    for i in range(len(y)):
        for s, x in zip(y[i], X[i]):
            if data.has_key(s):
                data[s].append(x)
            else:
                data[s] = [x]
    ns = len(data.keys())
    means = np.array([[np.mean(data[s])] for s in range(ns)])
    covars = np.tile(np.identity(1), (ns, 1, 1))
    for s in range(ns):
        covars[s, 0] = np.std(data[s])
    return mm.startprob, mm.transmat, means, covars

# lables 
label = {0: "Seattle", 1: "Boston", 2: "Washington D.C.", 3: "Philapedia", 4: "New York City"}
x = [[0 for i in range(10)] for j in range(1000)]#holds consumption
y = [[0 for i in range(10)] for j in range(1000)]#holds city
#formates the data in the tranning file into proper json fomate
f = open("json_formated.txt", "w")
for line in open("HW9_Q4_training.txt", "r").readlines():
    new_line = ""
    for i in range(len(line)):
        if line[i] == '\'':
            new_line += '\"'

        else:
            new_line += line[i]

    f.write(new_line) #writes the proper fomrated json to the new file

f.close()

i = 0
#loops through the file and intinalzes the tranning consuption and cities matrixs
for line in open("json_formated.txt", "r").readlines():
    entries = json.loads(line)
    j = 0
    for entry in entries:
        if entry[0] == "Seattle":
            y[i][j] = 0
            x[i][j] = entry[1]

        if entry[0] == "Boston":
            y[i][j] = 1
            x[i][j] = entry[1]

        if entry[0] == "Washington D.C.":
            y[i][j] = 2
            x[i][j] = entry[1]

        if entry[0] == "Philapedia":
            y[i][j] = 3
            x[i][j] = entry[1]

        if entry[0] == "New York City":
            y[i][j] = 4
            x[i][j] = entry[1]

        j += 1

    i += 1

#creastes a hinden markov model witht he tranning data
startprob, transmat, means, covars = estimate_parameters(x, y)
model = hmm.GaussianHMM(5, "full")
model.startprob_ = startprob
model.transmat_ = transmat
model.means_  = means
model.covars_ = covars

#l0ops through the testing data and writes to the predections file
f = open ("HW9_Q4_predictions.txt", "w")
for line in open("HW9_Q4_testing.txt", "r").readlines():
    formated_test = [[0 for i in range(1)] for j in range(10)]
    test = json.loads(line)
    #formates the array into a 2d array so it can be predicted
    for i in range(len(test)):
        formated_test[i][0] = test[i]

        #predicts the cities bassed on consumtion
    y = model.predict(formated_test)
    pred_lables = [0 for i in range(10)]
    i = 0
    #gets the lables for the data preidicted
    for s in y:
        pred_lables[i] = label[s]
        i += 1

    #writes to the file
    json_string = json.dumps(pred_lables)
    f.write(json_string)
    f.write("\n")