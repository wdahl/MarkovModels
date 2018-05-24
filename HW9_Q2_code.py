# William Dahl
# 001273655
# ICSI431 Data mining
# May 2nd, 2018
# coding: utf-8

# In[3]:

import numpy as np
from hmmlearn import hmm
import types
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


# In[32]:

label = {0: "Albany", 1: "Boston", 2: "Washinton D.C.", 3: "Philadelphia", 4: "New York City"}
X = [[[60.44],  [139.67], [231.84], [346.4], [18.82]],
     [[-200.04],  [52.7], [537.1], [109.82], [142.08]],
     [[79.09],  [-195.59], [177.49], [620.13], [100.25]],
     [[185.67], [254.93],  [62.12], [200.59], [152.77]],
     [[30.4], [180.69], [-336.63], [472.77], [36.17]],
     [[395.82], [346.42], [222.59], [213.88], [496.2]],
     [[95.53], [182.25], [3.76], [93.81], [159.14]],
     [[292.77], [158.31], [-101.1], [214.96], [-297.47]],
     [[129.03], [97.44], [369.64], [154.91], [61.69]],
     [[46.59], [-22.4], [-270.73], [41.39], [75.75]]]
y = [[0, 2, 3, 1, 0],
     [0, 4, 2, 4, 0],
     [1, 0, 3, 1, 4],
     [0, 2, 1, 0, 2],
     [0, 3, 0, 2, 4],
     [0, 2, 0, 3, 1],
     [1, 0, 4, 3, 0],
     [1, 0, 2, 3, 1],
     [4, 3, 1, 3, 2],
     [3, 4, 0, 4, 3]]


# Task 1: Predict the latent cities of training sequences

startprob, transmat, means, covars = estimate_parameters(X, y)
model = hmm.GaussianHMM(5, "full")
model.startprob_ = startprob
model.transmat_ = transmat
model.means_  = means
model.covars_ = covars


#Task 1: Suppose we have testing seqeunces of spendings, we now predict the latent cities of these sequences

X = [[[355]], [[339]], [[148]], [[50]]]

for x in X:
   y = model.predict(x)
   print [label[s] for s in y]

# Task 2: Let us predict the future three subsequent cities to be visited based on the following sequence of observations:<br/>
# [[355], [339], [148], [50]]

x = [[355], [339], [148], [50]]
y = hmm_predict_further_states(model, x, 3)
print [label[s] for s in y]


# Task 2: Let us predict the future three subsequent spendings based on the following sequence:
# [[355], [339], [148], [50]]

x = [[355], [339], [148], [50]]
cons = hmm_predict_future_features(model, x, 3)
print 7, [round(con[0], 2) for con in cons]