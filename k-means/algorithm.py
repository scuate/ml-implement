## k-means algorithm
import numpy as np
from numpy import log,float
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math

##assign data points to centers
def assignCtr(data,cs):
    asgn = {}
    for i in range(cs.shape[0]):
            asgn[i]= []
    dist_arr = cdist(data,cs,'euclidean') 
    for j in range(data.shape[0]):
        c_min = np.argmin(dist_arr[j], axis=0)
        asgn[c_min].append(data[j])
    return asgn  

##compute the new centers with the data points assigned to each center
def getCtr(data,asgn):
    ctrs = []
    for v in asgn.itervalues():
        if len(v)>0:   ##discard centers with no assignments
            pts = np.array(v)
            c_new = np.mean(pts, axis=0)
            ctrs.append(c_new)

    ctrs = np.array(ctrs)
    return ctrs

##farthest-first traversal
def FFT(ctrs,k):
    rd = np.random.randint(ctrs.shape[0],size=1)
    c1 = ctrs[rd[0]]
    ctr_lst = ctrs.tolist()
    ctr_lst.remove(c1.tolist())
    ctrnew = [c1]
    for i in range(1,k):
        dist_arr = cdist(np.array(ctr_lst), ctrnew,'euclidean')
        farthest = np.argmax(dist_arr.sum(axis=1), axis=0)
        ctrnew.append(ctr_lst.pop(farthest))
        i += 1
    return np.array(ctrnew)

##power initialization
def PI(data,k,c):
    n = data.shape[0]
    k_p = c*k*log(k)
    rd = np.random.randint(n,size=k_p)
    cs = data[rd]
    asgn = assignCtr(data, cs)
    ctrs = getCtr(data, asgn)
    asgn_new = assignCtr(data, ctrs)
    ctr_lst = ctrs.tolist()
    
    for ck,v in asgn_new.iteritems():
        if len(v) < n/float(math.e*k_p):
            ctr_lst.remove(ctrs.tolist()[ck]) ##ctr_lst is changing,so the removed item should be from orig array
    ctrs = np.array(ctr_lst)
    ctrs_new = FFT(ctrs,k)
     
    return ctrs_new

##naive initialization(to compare with power initialization)
def NI(data,k):
    n = data.shape[0]
    rd = np.random.randint(n,size=k)
    cs = data[rd]
    return cs

##run k-means to get the cluster centers
def kmeans(data,k,cs,T):
    iterations = 1
    ctrs = cs
    prev_ctrs = None
    while (not np.array_equal(ctrs, prev_ctrs)) and (iterations<=T):
        prev_ctrs = ctrs
        iterations += 1
        asgn = assignCtr(data, ctrs)
        ctrs = getCtr(data,asgn)
    return ctrs

##added the parameters--colour,initial--for clear plotting and comparison
def kmeans_plot(data,k,cs,T,colour,initial):
    iterations = 1
    ctrs = cs
    prev_ctrs = None
    while (not np.array_equal(ctrs, prev_ctrs)) and (iterations<=T):
        x = ctrs[:,0]
        y = ctrs[:,1]
        plt.plot(x,y, "ro", color = colour,label=initial if iterations==1 else "")
        prev_ctrs = ctrs
        iterations += 1
        asgn = assignCtr(data, ctrs)
        ctrs = getCtr(data,asgn)

##compute the total cost(sum of euclidean distance squared between data points and cluster centers)
def kmeans_cost(asgn,cs):
    cost = 0
    for i in range(cs.shape[0]):
        dist_arr = cdist(asgn[i], np.array([cs[i]]),'euclidean')
        dist_arr = dist_arr**2
        cost += np.sum(dist_arr,axis=0)
    return log(cost)  
        
##added the parameters--colour,initial--for clear plotting and comparison
def cost_plot(data,k,cs,T,colour,initial):
    iterations = 1
    ctrs = cs
    prev_ctrs = None
    while (not np.array_equal(ctrs, prev_ctrs)) and (iterations<=T):
        asgn = assignCtr(data, ctrs)
        lgcost = kmeans_cost(asgn, ctrs)
        plt.plot(iterations,lgcost, ".", color = colour,label=initial if iterations==1 else "")
        prev_ctrs = ctrs
        iterations += 1      
        ctrs = getCtr(data,asgn)







