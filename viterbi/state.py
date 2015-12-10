##the State class used by viterbi.py
from sys import argv
import math


##a class of the State
class State():
    def __init__(self,tag,emis_prob):
        self.previous = None  ##the back pointer
        self.prevlst = []  ##a list of all prev states for this state with transition probs
        self.tag = tag  ##tag name of the state
        self.emis = emis_prob  ##emission prob P(w|t) of this state
        self.lgprob = 0.0  ##max lgprob after computation
      
    ##add a prev state with transition prob to the prevlst
    def addPrevTup(self,prev,tran):
        self.prevlst.append((prev,tran))
    
    ##get the maxprob previous state stored in the back pointer
    def getPrev(self):
        return self.previous
    
    ##compute with prevLst the path from a previous state with max prob, store the previous state in the back pointer
    def cmptLgprob(self): 
        maxprob = float("-inf")
        maxprev = None
        for prev,tran in self.prevlst:
            prev_prob = prev.getLgprob()
            total_tran = tran + prev_prob
            if total_tran > maxprob:
                maxprob = total_tran
                maxprev = prev
        self.lgprob = maxprob + self.emis
        self.previous = maxprev
            
    
    def addLgprob(self,logprob):
        self.lgprob = logprob
     
    ## get the lgprob stored in the state   
    def getLgprob(self):
        return self.lgprob
    
    ## get the tag name
    def getTag(self):
        return self.tag
        
