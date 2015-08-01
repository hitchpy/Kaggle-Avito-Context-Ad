import numpy as np
import pandas as pd
from math import sqrt, exp, log
import sqlalchemy as sa
import ast
from datetime import datetime
from random import random

class ftrl(object):
    def __init__(self, alpha, beta, l1, l2, bits):
        self.z = [0.] * bits
        self.n = [0.] * bits
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.w = {}
        self.X = []
        self.y = 0.
        self.bits = bits
        self.Prediction = 0.

    def sgn(self, x):
        if x < 0:
            return -1  
        else:
            return 1

    def fit(self,line):
        try:
            self.ID = line['TestId']
            del line['TestId']
        except:
            pass

        try:
            self.y = float(line['IsClick'])
            del line['IsClick']
        except:
            pass
        
        del line['SearchID']
        del line['AdID']
        del line['ObjectType']
        for k in line.keys():
            if isinstance(line[k], basestring):
                line[k] = line[k]
            else:
                line[k] = str(line[k])       
        self.X = [0.] * len(line)
        for i, key in enumerate(line):
            val = line[key]
            if isinstance(key, basestring):
                self.X[i] = (abs(hash(key + '_' + val)) % self.bits)
            else:
                self.X[i] = (abs(hash(str(key) + '_' + val)) % self.bits)
        self.X = [0] + self.X

    def logloss(self):
        act = self.y
        pred = self.Prediction
        predicted = max(min(pred, 1. - 10e-15), 10e-15)
        return -log(predicted) if act == 1. else -log(1. - predicted)

    def predict(self):
        W_dot_x = 0.
        w = {}
        for i in self.X:
            if abs(self.z[i]) <= self.l1:
                w[i] = 0.
            else:
                w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + sqrt(self.n[i]))/self.alpha) + self.l2)
            W_dot_x += w[i]
        self.w = w
        self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))
        return self.Prediction

    def update(self, prediction): 
        for i in self.X:
            g = (prediction - self.y) #* i
            sigma = (1./self.alpha) * (sqrt(self.n[i] + g*g) - sqrt(self.n[i]))
            self.z[i] += g - sigma*self.w[i]
            self.n[i] += g*g

def clean(line):
    t = line["SearchDate"]
    t =  datetime.strptime(t, "%Y-%m-%d %H:%M:%S.0")
    line["SearchDate"] = t.month*30 + t.day
    line["hour"] = t.hour
    
    title = set(line["Title"].split(" "))
    if line['SearchQuery'] == '':
        line['SearchQuery'] = -1
    else:
        a = set(line['SearchQuery'].split(' '))
        line['SearchQuery'] = len(a.intersection(title))
    del line["Title"]
    for t in title:
        line[t] = ""
    return line
        
            
if __name__ == '__main__':
    """
    SearchID	AdID	Position	ObjectType	HistCTR	IsClick
    """
    engine = sa.create_engine('sqlite:///../input/database.sqlite')
    #logs = open('log.txt','w+')
    clf = ftrl(alpha = 0.2, 
               beta = 1., 
               l1 = 1.,
               l2 = 1.0, 
               bits = 2**25)

    loss = 0.
    count = 0
    #query = 'SELECT * FROM trainSearchStream, 100000'
    query = "SELECT trainSearchStream.*, AdsInfo.CategoryID AS acid, IPID, Price, Title, SearchDate, IsUserLoggedOn, SearchQuery, SearchInfo.LocationID, SearchInfo.CategoryID, UserAgentOSID, UserDeviceID FROM trainSearchStream LEFT JOIN SearchInfo ON trainSearchStream.SearchID=SearchInfo.SearchID LEFT JOIN AdsInfo ON AdsInfo.AdID=trainSearchStream.AdID LEFT JOIN Category c1 ON SearchInfo.CategoryID=c1.CategoryID LEFT JOIN UserInfo ON UserInfo.UserID=SearchInfo.UserID LEFT JOIN Category c2 ON AdsInfo.CategoryID=c2.CategoryID WHERE ObjectType=3;" 
    stream = pd.read_sql(query, engine, chunksize = 1000)
    for chunk in stream:
        for i in range(chunk.shape[0]):
            line = dict(chunk.loc[i,:])
            if line['IsClick'] == 0:
                if random() < 0.3:
                    count += 1
                    continue
            line = clean(line)
            clf.fit(line)
            pred = clf.predict()
            loss += clf.logloss()
            clf.update(pred)
            count += 1
            if count%10000 == 0: 
                #logs.write("(seen {}, loss {}".format(count, loss * 1./count) + "\n")
                print "seen {}, loss {}".format(count, loss * 1./count)

    #query = 'SELECT * FROM testSearchStream 100000'
    query = "SELECT testSearchStream.*, AdsInfo.CategoryID AS acid, IPID, Price, Title, SearchDate, IsUserLoggedOn, SearchQuery, SearchInfo.LocationID, SearchInfo.CategoryID, UserAgentOSID, UserDeviceID FROM testSearchStream LEFT JOIN SearchInfo ON testSearchStream.SearchID=SearchInfo.SearchID LEFT JOIN AdsInfo ON AdsInfo.AdID=testSearchStream.AdID LEFT JOIN Category c1 ON SearchInfo.CategoryID=c1.CategoryID LEFT JOIN UserInfo ON UserInfo.UserID=SearchInfo.UserID LEFT JOIN Category c2 ON AdsInfo.CategoryID=c2.CategoryID WHERE ObjectType=3;"
    stream = pd.read_sql(query, engine, chunksize = 5000)
    count = 0
    with open("submission7.csv", 'w') as outfile:
        outfile.write('ID,IsClick\n')
        for chunk in stream:
            count += 5000
            for i in range(chunk.shape[0]):
                line = dict(chunk.loc[i,:])
                line = clean(line)
                clf.fit(line)
                p = clf.predict()
                outfile.write('%s,%s\n' % (clf.ID, p))
            if count %10000 == 0:
                print "Finished %d predictions" %count
            
