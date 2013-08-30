# ----------- Meme.py --------------
# this file contains the class definition for a meme, which will contain
# our representation of one.


from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk import ngrams
from nltk.classify import *
from nltk import PorterStemmer 

from collections import defaultdict
from random import shuffle
import operator
import math
import sys
import csv
import pickle


class Meme:

	#####[ --- Data --- ] #####
	parent = None
	meme_type = None

	#--- raw text ---
	top_text_raw = None
	bottom_text_raw = None
	all_text_raw = None

	#--- tokenized text ---
	top_text = None
	bottom_text = None
	all_text = None

	#--- feature representations ---
	ngram_features = {}
	tfidf_features = {}
	sentiment_features = {}



	# Function: Constructor
	# ---------------------
	# given the raw top/bottom text, this will fill in all the data.
	def __init__(self, parent, meme_type, top_text_raw, bottom_text_raw):

		### Step 1: fill in basic data ###
		self.parent = parent
		self.meme_type = meme_type
		self.top_text_raw = top_text_raw
		self.bottom_text_raw = bottom_text_raw
		self.all_text_raw = self.top_text_raw + " | " + self.bottom_text_raw

		### Step 2: get tokenized versions ###
		self.tokenize ();






	# Function: tokenize
	# ------------------
	# self.(top|bottom)_text_raw -> self.(top|bottom|all)_text
	def tokenize (self):

		top_sentences = word_tokenize (self.top_text_raw)
		bottom_sentences = word_tokenize (self.bottom_text_raw)

		top_tokenized_sentences = [word_tokenize (s) for s in top_sentences]
		bottom_tokenized_sentences = [word_tokenize (s) for s in bottom_sentences]

		self.top_text = []
		for s in top_tokenized_sentences:
			for word in s:
				self.top_text.append (word)

		self.bottom_text = []
		for s in bottom_tokenized_sentences:
			for word in s:
				self.bottom_text.append(word)

		self.all_text = self.top_text + self.bottom_text



	# Function: get_ngram_features 
	# ----------------------------
	# fills in ngram_features with a dict of stemmed_word:True pairs. 
	# (NOTE: currently only unigrams. Add in common bigrams?)
	def get_ngram_features (self):

		stemmer = PorterStemmer ()

		top_features = [(stemmer.stem(token) + "__TOP__", True) for token in self.top_text]
		bottom_features = [(stemmer.stem(token) + "__BOTTOM__", True) for token in self.bottom_text]
		all_features = [(stemmer.stem(token) + "__ALL__", True) for token in self.all_text]
		self.ngram_features = dict(top_features + bottom_features + all_features);



	# Function: get_features
	# ----------------------
	# returns a feature vector representation of this meme
	# since we are training maxent, it will be a dict of word:True pairs
	def get_features (self):

		self.get_ngram_features ()
		return self.ngram_features


	# Function: string representation
	# -------------------------------
	# a string representation of the meme 
	def __str__ (self):

		return self.meme_type + ": " + self.top_text_raw + " / " + self.bottom_text_raw

















