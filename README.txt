Text Classification of Political Blogs

#Task: given the training data of political blogs, predict the political opinion of each blog in the test data (left-leaning or right-leaning)

#Datasets: Training/left:3184,right:3256 
	   Test/left:399,right:407

#Model: Maximum Entropy(using Mallet)

#Features:
F1-unigram:raw frequency in a doc 
F2-bigram:raw frequency in a doc
F3-trigram:raw frequency in a doc
F4-count of party members(extracted from an external source) in a doc divided by total word count in a doc
F5-count of party keywords(extracted from an external source) in a doc divided by total word count in a doc
F6-for all ngrams used as features, replace the rawFreq with 0.5+0.5*rawFreq/maxWordFreq_in_doc to avoid bias towards longer docs

#Baseline(of test accuracy):
F1 only -    	93.6725%
#Results:
F1+F2 -		94.7891%
F1+F3 - 	93.6725%
F1+F4 - 	93.7965%
F1+F5 - 	94.0447%
F1+F6 - 	94.2928%
F1+F2+F6 - 	96.0298% -> best result 

#File Structure
./data/ contains sample training and test dir, external source data extracted from webpages
./scraper/ contains python scripts for extracting information from webpages
./expt/ contains sample program output files (training/test vector files and MaxEnt classification results)
./train_classiry.sh the wrapper for creating feature vectors, running the classifier and writing results
./create_vectors.sh the shell script to invoke create_vectors.py
./create_vectors.py create feature vectors for all files under given dir
./proc_file.sh the shell script to invoke create_vectors.py with option '-f' to process a single input doc
./proc_file.py called by create_vectors.py to process an individual file

#Usage
1. to process a single file, output the feature vectors to a give file:

./proc_file.sh input_file targetLabel output_file

2. to process all files under multiple dirs, output the feature vectors to a given file:

./create_vectors.sh output_vector_file dir1 dir2 ... dir_n

3. to create feature vectors for all the training and test data, run the classifier and output all the intermediate and resultative files to a given dir:

./train_classify.sh param_file train_dir test_dir output_dir
