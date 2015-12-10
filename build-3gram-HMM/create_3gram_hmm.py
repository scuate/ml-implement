##create an Hidden Markov Model for a trigram POS tagger
from sys import stdin,argv
from ngram import Ngram
import math
import re

##read in the training data (format-"word1/tag1 word2/tag2..."),make ngram objects with info of tag ngram counts and tag-word coocurrence counts
def readFile():
    uni_tag = Ngram()
    bi_tag = Ngram()
    tri_tag = Ngram()
    for s in stdin:
        tags = []
        words = []
        if s != '' and s != '\n':
            s = '<s>/BOS <s>/BOS '+s.rstrip()+' </s>/EOS'
            lst = s.split()
            for wt in lst:  #parse words and tags
                t = wt.split('/')[-1]
                tags.append(t)
                idx = wt.rfind(t)
                w = wt[:idx-1]
                words.append(w)
            for i in range(len(tags)):#go through the training data,add counts of tag ngrams and tag-word co-occurences
                t = tags[i]
                w = words[i]
                uni_tag.addEntry(t,w)
                if i <= len(tags)-2:
                    seq = ' '.join(tags[i:i+2])
                    bi_tag.addCount(seq)
                    if i <= len(tags)-3:
                        seq2 = ' '.join(tags[i:i+3])
                        tri_tag.addCount(seq2)
    return uni_tag,bi_tag,tri_tag

##read the file of tags with unknown-word probabilities,store tag-prob in a dict    
def readUnk(unkf):
    dic = {}
    for line in unkf:
        line = line.rstrip()
        lst = line.split()
        tag = lst[0]
        prob = lst[1]
        dic[tag] = float(prob)
    return dic

##get the values from ngram dicts to calculate the number of states, symbols,transmission, and emission for the header
def cmptVar(uni_tag,bi_tag,tri_tag):
    uni_dic = uni_tag.dic
    bi_dic = bi_tag.count
    tri_dic = tri_tag.count
    tagNum = float(uni_tag.getTagnum())
    entNum = float(uni_tag.getTotal())
    state_num = (int(tagNum)-1)*int(tagNum) - (int(tagNum)-2)
    sym_num = uni_tag.getWordnum()
    trans_line_num = int((tagNum-1)*(tagNum-2)*(tagNum-1)) + int(tagNum) - 1
    emiss_line_num = (uni_tag.getWordnum() + int(tagNum))*(int(tagNum)-1)- (len(uni_dic['BOS'])+1)*(int(tagNum)-2) ##considering t1 can only be BOS when t2 is BOS, len(uni_dic['BOS'])+1 because there's unknown word
    return uni_dic,bi_dic,tri_dic,tagNum,entNum,state_num,sym_num,trans_line_num,emiss_line_num


##write the header and the init prob of the output hmm file
def writeHeader(outfile,state_num,sym_num,trans_line_num,emiss_line_num)
    outfile.write('state_num={0}\n'.format(state_num))
    outfile.write('sym_num={0}\n'.format(sym_num))
    outfile.write('init_line_num={0}\n'.format(1))
    outfile.write('trans_line_num={1}\nemiss_line_num={2}\n\n'.format(trans_line_num,emiss_line_num))
    outfile.write('\\init\n{0:<8}{1:<8}{2:<8}\n\n\n'.format('BOS_BOS',1.0,0.0))

##consider all POSSIBLE permutations of t1,t2,t3 sequences(POSSIBLE means exclusion of EOS EOS word,etc.),compute P(t3|t1,t2) = l1*P1(t3)+l2*P2(t3|t2)+l3*P3(t3|t1,t2). P(t3|t1,t2) is represented as the transition prob between t1_t2 and t2_t3-- the from-state is t1_t2; the to-state is t2_t3
def writeTransit(outfile,tagNum,uni_dic,bi_dic,tri_dic):
    outfile.write('\\transition\n')
    tagset = tagNum - 2.0

    key_t1t2 = uni_dic.keys() #the values t1,t2 can assume
    key_t1t2.remove('EOS') #t1,t2 can't be 'EOS'
    for t3 in uni_dic:
        if t3 != 'BOS': #t3 can't be 'BOS'
            p1 = uni_tag.getCount(t3)/(entNum-uni_tag.getCount('BOS'))##t3 can't be 'BOS'
            for t2 in key_t1t2:
                p2 = 0.0
                bi = ' '.join((t2,t3))
                if bi in bi_dic:
                    if t2 == 'BOS':
                        p2 = bi_tag.getCount(bi)/(float(uni_tag.getCount(t2))-float(bi_tag.getCount('BOS BOS')))##t3 can't be 'BOS', so deduct the double BOS cases from the denominator
                    else:
                        p2 = bi_tag.getCount(bi)/float(uni_tag.getCount(t2))
                if t2 == 'BOS':
                    t1 = 'BOS'  ##if t2==BOS, t1 can only be BOS
                    p3 = 0.0
                    tri = ' '.join((t1,t2,t3))
                    prev = ' '.join((t1,t2)) 
                    if tri in tri_dic:
                        p3 = tri_tag.getCount(tri)/float(bi_tag.getCount(prev))
                    p = l1*p1 + l2*p2 + l3*p3
                    lg_p = math.log10(p)
                    from_st = t1+'_'+t2
                    to_st = t2+'_'+t3
                    outfile.write('{0:<15}{1:<15}{2:<20}{3:<20}\n'.format(from_st,to_st,p,lg_p))
                    continue   
                for t1 in key_t1t2:
                    p3 = 0.0                   
                    tri = ' '.join((t1,t2,t3))
                    prev = ' '.join((t1,t2))                    
                    if prev not in bi_dic:
                        p3 = 1.0/(tagset+1.0)
                    elif tri in tri_dic:
                        p3 = tri_tag.getCount(tri)/float(bi_tag.getCount(prev))                                      
                    p = l1*p1 + l2*p2 + l3*p3 #use linear interpolation for the trigram transition probs               
                    lg_p = math.log10(p)
                    from_st = t1+'_'+t2
                    to_st = t2+'_'+t3
                    outfile.write('{0:<15}{1:<15}{2:<20}{3:<20}\n'.format(from_st,to_st,p,lg_p))


##compute the emission probs of words for all POSSIBLE permutations of (t1,t2 to get states t1_t2(e.g. no 'EOS''BOS'). Only t2 emits the word, but t1 is encoded in the state to correspond with the states in the transition model.Smooth the emission probs by inroducing P(<unknown>|tag) read from the input file. For all seen words,P(w|tag) = cnt(w,tag)/cnt(tag)*(1-P(<unknown>|tag))
def writeEmission(outfile,unkf,uni_dic,bi_dic,tri_dic):
    outfile.write('\n\\emission\n')
    unk = readUnk(unkf) ##read in the probs of tags associated with unknown words
    for t1 in key_t1t2: 
        for t2 in uni_dic:
            if t2 == 'BOS' and t1 != 'BOS':
                continue
            st = t1+'_'+t2  ##compose the state
            tc = float(uni_tag.getCount(t2))
            p_unk = 0.0
            if t2 in unk:
                p_unk = unk[t2]
            for w in uni_dic[t2]:  ##only t2 emits the word
                wc = float(uni_dic[t2][w])
                p = wc/tc
                p_smooth = p*(1.0-p_unk)
                lg_p = math.log10(p_smooth)
                outfile.write('{0:<15}{1:<25}{2:<20}{3:<20}\n'.format(st,w,p_smooth,lg_p))
            if p_unk != 0.0: 
                outfile.write('{0:<15}{1:<25}{2:<20}{3:<20}\n'.format(st,'<unk>',p_unk,math.log10(p_unk)))
            else:
                outfile.write('{0:<15}{1:<25}{2:<20}{3:<20}\n'.format(st,'<unk>',0.0,'-inf'))

##write to the outfile in the required format w/ header and end       
def write_hmm(argv,ntags):
    outfile = open(argv[1],'w')
    l1 = float(argv[2])
    l2 = float(argv[3])
    l3 = float(argv[4])
    unkf = open(argv[5])
    uni_tag = ntags[0]
    bi_tag = ntags[1]
    tri_tag = ntags[2]

    uni_dic,bi_dic,tri_dic,tagNum,entNum,state_num,sym_num,trans_line_num,emiss_line_num = cmptVar(uni_tag,bi_tag,tri_tag)

    writeHeader(outfile,state_num,sym_num,trans_line_num,emiss_line_num)

    writeTransit(outfile,tagNum,uni_dic,bi_dic,tri_dic,l1,l2,l3)

    writeEmission(outfile,unkf,uni_dic,bi_dic,tri_dic)
       
    
if __name__ == "__main__":
    ntags = readFile()
    write_hmm(argv,ntags)
