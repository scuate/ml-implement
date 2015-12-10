##a class imported by create_3gram_hmm.py
from sys import argv
import math


##a class of ngram groups, with info such as tag ngram counts,tag-word co-occurences count, total number of tags, total number of words
class Ngram():
    def __init__(self):
        self.dic = {}
        self.count = {}
        self.tagNum = 0
        self.wordNum = 0
        self.total = 0
    
    def addCount(self,ngram):
        if ngram not in self.count:
            self.count[ngram] = 1
            self.tagNum += 1
        else:
            self.count[ngram] += 1
    
    def addEntry(self,tag,word):
        self.total += 1
        if tag not in self.dic:
            self.count[tag] = 1
            self.dic[tag] = {word:1}
            self.tagNum += 1
            self.wordNum += 1
        else:
            self.count[tag] += 1
            if word in self.dic[tag]:
                self.dic[tag][word] += 1
            else:
                self.dic[tag][word] = 1
                self.wordNum += 1
        
    def getCount(self,ngram):
        return self.count[ngram]
    
    def getDic(self):
        return self.dic
    
    def getWordnum(self):
        return self.wordNum
    
    def getTagnum(self):
        return self.tagNum
    
    def getTotal(self):
        return self.total
