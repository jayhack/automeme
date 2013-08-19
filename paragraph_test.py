#!/usr/bin/python
#---------
# File: paragraph_test.py
# -----------------------
# trying to get it to generate memes/captions for every line in a paragraph
#
import os
import sys
from collections import defaultdict
from SentimentAnalysis import SentimentAnalysis



if __name__ == "__main__":

	### Step 1: load the classifier ###
	filenames = defaultdict(lambda: None)
	filenames['ngrams_classifier'] = "ngrams_classifier.obj"
	filenames['meme_types'] = "meme_types.txt"
	save = False
	sa = SentimentAnalysis (filenames, save, mode='use_mode')


	### Step 2: apply it to a test sentence ###
	sorted_prob_dist = sa.classify_sentence ("this is just a test");
	print sorted_prob_dist