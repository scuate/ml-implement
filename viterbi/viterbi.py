##use viterbi algorithm to compute the best tag sequences for the input sentences based on the HMM with states like t1_t2

from sys import argv
from state import State
import math

##read in the hmm, store transition and emission probs in respective dictionaries
def readHmm(argv):
    f = open(argv[1])
    header = {}
    line = f.readline().rstrip()
    while not '\\init' in line:
        line = f.readline().rstrip()
    line = f.readline().rstrip()
    init_prob = None
    while not '\\transition' in line:
        if line != '':
            lst = line.rstrip().split()
            init_prob = float(lst[2])
        line = f.readline().rstrip()
    line = f.readline().rstrip()
    tran = {}  #a dict that maps from_st:[to_st1,to_st2], iterate over the list for all possible next states
    reverse_tran = {}  #a nested dict,reverse_tran[to_st][from_st]=trans_prob. While at to_st,get all the trans_prob from prev states easily
    while not '\\emission' in line:
        if line != '':
            lst = line.rstrip().split()
            from_st = lst[0]
            to_st = lst[1]
            log_p = float(lst[3])
            if from_st not in tran:
                tran[from_st] = [to_st]
            else:
                tran[from_st].append(to_st)
            if to_st not in reverse_tran:
                reverse_tran[to_st] = {from_st:log_p}
            else:
                reverse_tran[to_st][from_st] = log_p
        line = f.readline().rstrip()
    line = f.readline().rstrip()
    reverse_emis = {}  ##a nested dict, reverse_emis[word][state]=log_p. Reverse mapping allows considering a limited number of states when reading in a word from the input
    while line != '':
        lst = line.rstrip().split()
        st = lst[0]
        w = lst[1]
        p = float(lst[2])
        log_p = float(lst[3])
        if p != 0.0:
            if w not in reverse_emis:
                reverse_emis[w] = {st:log_p}
            else:
                reverse_emis[w][st] = log_p
        line = f.readline().rstrip() 
    f.close()
    return (init_prob,tran,reverse_tran,reverse_emis)


##a method to be called by viterbi() to output tag sequences
def writeTags(outfile,line,maxstate,maxprob):
    cur_state = maxstate
    cur_tag = maxstate.getTag()
    lst = [cur_tag]
    while cur_tag != 'BOS_BOS': #keep calling the back pointer until reaching the first object
        cur_state = cur_state.getPrev()
        cur_tag = cur_state.getTag()
        lst.insert(0,cur_tag)
    outseq = ' '.join(lst)
    outfile.write('{0} => {1}  {2}\n'.format(line,outseq,maxprob))

##viterbi algorithm: for each possible state at time T, compute the path from a previous state at time T-1 with the max prob, put that previous state in the back pointer; at the end of the input sequence, pick the final state with max prob, and read the back pointer until at the beginning of the input seq    
def viterbi(argv,hmm):
    init_prob = hmm[0]
    tran = hmm[1]  ##the dict of tran[from_st]=[to_st]
    rtran = hmm[2] ##the dict of reverse_tran[to_st][from_st]=tran_prob
    emis = hmm[3] ##the dict of reverse_emis[w][state]=emiss_prob
    f = open(argv[2])
    outfile = open(argv[3],'w')
    for line in f:
        line = line.rstrip()
        lst = line.split()
        init_state = State('BOS_BOS',0.0) ##start the trellis with init_state
        init_state.addLgprob(0.0)
        prev_states = [init_state]
        for i in range(len(lst)):
            if lst[i] in emis:##use reverse_emis dict to get all states that could emit this word
                emis_states = set(emis[lst[i]].keys())
            else:  ##for unseen words,get all states that emit <unk>
                emis_states = set(emis['<unk>'].keys())

            cur_states = []#a list of State objects at current T(idx correspond with cur_tags)
            cur_tags = []#a list of tags at cur T
            for from_st in prev_states:
                from_tag = from_st.getTag()
                to_tags = set(tran[from_tag])
                to_tags = list(to_tags.intersection(emis_states))##intersection of possible next states by transition and emission->a limited number of next states
                if len(to_tags) > 0:
                    for to_tag in to_tags:
                        if to_tag not in cur_tags: #avoid redundant tags transitioned from different prev tags
                            if lst[i] in emis and to_tag in emis[lst[i]]:
                                emis_prob = emis[lst[i]][to_tag]
                            else:
                                emis_prob = emis['<unk>'][to_tag]
                            st = State(to_tag,emis_prob)  ##initialize each state
                            tran_prob = rtran[to_tag][from_tag]
                            st.addPrevTup(from_st,tran_prob) ##add a candidate for the prev T
                            cur_tags.append(to_tag)
                            cur_states.append(st)
                        else:  #if the tag exists,add the candidate to the existing State
                            idx = cur_tags.index(to_tag)
                            tran_prob = rtran[to_tag][from_tag]
                            cur_states[idx].addPrevTup(from_st,tran_prob)

            for st in cur_states: 
                st.cmptLgprob() ##compute the path from a previous state with max prob, store the previous state in the back pointer--see state.py
            prev_states = cur_states ## store all possible current states of time T to previous states for time T+1
            
        maxprob = float("-inf")
        maxstate = None
        ## pick the final state with max prob
        for state in cur_states:
            final_prob = state.getLgprob()
            if final_prob > maxprob:
                maxprob = final_prob
                maxstate = state
        writeTags(outfile,line,maxstate,maxprob)

    f.close()
    outfile.close()
                
                
    
if __name__ == "__main__":
    hmm = readHmm(argv)
    viterbi(argv,hmm)
