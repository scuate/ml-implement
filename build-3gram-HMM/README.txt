A Hidden Markov Model for a Trigram POS Tagger

The transition prob P(t3|t1,t2) is computed by linear interpolation, i.e. P(t3|t1,t2)=l1*P1(t3)+l2*P2(t3|t2)+l3*P3(t3|t1,t2). In the output HMM file, P(t3|t1,t2) is represented by the transition prob between two states-- t1_t2 and t2_t3

The emission prob P(w|tag) is smoothed by introducing P(<unknown_word>|tag). P(w|tag)=cnt(w,tag)/cnt(tag)*(1-P(<unk>|tag)). Emission probs for all states t1_t2 are written to the HMM file. The states correspond with those in the transition model, even though the emission prob is between t2 and word -> P(w|t2)

-----
File structure:
./data/ contains sample input, sample output, the tag_unknown_prob file
create_3gram_hmm.py  reads in the training data and outputs the HMM model
ngram.py is a class used by create_3gram_hmm.py 
