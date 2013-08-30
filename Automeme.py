#!/usr/bin/python 
# ---------------------------------------------------------- #
# Class: Automeme.py
# -----------------
# contains everything pertaining to the classification of memes
# includes loading, training, classifying, etc
#
#
#
# ---------------------------------------------------------- #

#--- Standard ---
import os
import sys
import csv
import pickle
from collections import defaultdict
from random import shuffle
import operator
import math


#--- NLTK ---
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk import ngrams
from nltk import classify
from nltk import PorterStemmer 

#--- My files ---
from Meme import Meme 
from common_utilities import print_message, print_status, print_inner_status, print_error


#--- Globals ---
data_directory          = os.path.join (os.getcwd(), 'data')
saved_data_directory    = os.path.join (os.getcwd(), 'saved_data')
classifiers_directory   = os.path.join (os.getcwd(), 'classifiers')





# Class: Automeme
# ------------------------
# the main class for this program, will classify memes or sentences.
class Automeme:
    
    #---------------------[ Meme Examples ]--------------------------
    meme_examples       = []      # list of Meme objects representing all the examples we have access to
    meme_types          = {}    # defaultdict mapping meme_type (strings) to the count of examples we have of them

    #---------------------[ Training Data ]--------------------------
    all_examples        = []       # (features, meme_type) representation for all examples
    training_examples   = []       # (features, meme_type) representation for memes used to train classifier
    testing_examples    = []       # (featuers, meme_type) representation for memes used to evaluate classifier    

    #---------------------[ Classification ]-------------------------
    classifier = None






    #---------------------[ TF-IDF RELATED MECHANISMS ]---------------------
    stemmer = PorterStemmer ()
    vocab = set ()              # vocab[meme_type][c_name] = {set of all words that occur in meme_type under c_name}
    word_frequencies = {}       # word_frequencies[meme_type][c_name][word] = # of times word occurs in meme_type under c_name
    tfidf = {}                  # tfidf[meme_type][c_name][word] = tfidf of word in meme_type under c_name

    #---------------------[ NGRAMS CLASSIFIER ]---------------------
    ngrams_classifier = {}                      # classifier[c_name] = classifier for the c_name portion of memes



    ########################################################################################################################
    #########################[--- Constructor/Initialization ---]###########################################################
    ########################################################################################################################

    # Function: constructor
    # ---------------------
    # Initializes data structures, gets all labeled data, though does not train the classifiers.
    def __init__ (self):
       
        print_status ("Initialization", "Loading memes")
        # self.load_meme_examples_pickle ()

        print_status ("Initialization", "Getting feature represenations")
        # self.get_training_examples ()

        print_status ("Initialization", "Training classifier")
        # self.train_classifier ()
        self.load_classifier ('unigram_classifier.obj')

        # print_status ("Initialization", "Saving classifier")
        # self.save_classifier ("unigram_classifier.obj")

        #--- Evaluation ---
        # self.classifier.show_most_informative_features (500)
        # print "### total accuracy: ###"
        # print classify.accuracy (self.classifier, self.testing_examples)
        # MRR = self.evaluate_classifier_MRR ()
        # print "### MRR: ###"
        # print MRR






        # if mode == 'dev_mode':
        #     print_status ("Initialization", "Entering dev mode")
        #     self.save = save

        #     print_status ("Initialization", "Loading meme examples")
        #     self.load_meme_examples(filenames['meme_examples'])

        #     print_status ("Initialization", "Getting meme types")



        #     ### get labelled data from our list of memes and partition it for training ###
        #     print "TRAINING DATA:"
        #     self.get_ngrams_labeled_data ()
        #     self.partition_ngrams_labeled_data(0.85)
        #     print "     # of train examples = ", len(self.ngrams_train_data)
        #     print "     # of test examples = ", len(self.ngrams_test_data)




        #     ### get the classifier itself, via training or loading ###
        #     if filename['ngrams_classifier']:
        #         print "LOAD CLASSIFIER: (filename = " + filename + ")"
        #         f = open(filename, 'r')
        #         self.ngrams_classifier = pickle.load(f)
        #         f.close ()
        #     else:
        #         print "TRAIN CLASSIFIER:"
        #         self.train_ngrams_classifier (self.ngrams_train_data);


        #     print "EVALUATE CLASSIFIER:"
        #     self.evaluate_ngrams_classifier (self.ngrams_test_data);





        # # Mode: use_mode
        # # --------------
        # # loads the classifier and meme types; super quick, for real usage.
        # elif mode == 'use_mode':

        #     print "LOAD: memes types"
        #     self.load_meme_types (filenames['meme_types'])

        #     self.get_ngrams_classifier (self.ngrams_train_data, filenames['ngrams_classifier'])



        # print "Geting vocab/word frequencies... "
        # self.load_vocab_and_word_frequencies (filenames['vocab'], filenames['word frequencies'])

        # print "Getting tf.idf..."
        # self.load_tfidf (filenames['tfidf'])

        # print "Loading sentiment lexicon data..."
        # self.sentiment_lexicon_manager = SentimentLexiconManager (filenames['sentiment'])

        # print "Converting memes to labeled feature vectors... CURRENTLY TFIDF, GOTTA CHANGE THIS"
        # self.get_labeled_data ()
        # self.get_labeled_data_tfidf ()

        # print "Partitioning data into training and test sets for cross-validation..."
        # self.partition_data (0.85)
        
        #print "Loading classifier(s)..."
        #self.load_maxent (filenames);

        return





    ########################################################################################################################
    ########################[ --- Loading/Saving/Initializing Memes --- ]###################################################
    ########################################################################################################################

    # Function: (load|save)_meme_examples_(text|pickle)
    # ----------------------------
    # this function will load all examples of memes we have into objects of type 'Meme' 
    # (see Meme.py) and store them in self.meme_examples
    #
    # File format:
    # ------------
    # meme_type | top_text | bottom_text
    #
    def load_meme_examples_text (self):
    
        self.meme_types = {}
        self.meme_examples = []


        meme_filenames = [os.path.join (data_directory, filename) for filename in os.listdir(data_directory)]
        for filename in meme_filenames:

            print_inner_status ("Loading file", filename)
            f = open(filename, 'r')
            entries = [m for m in f.readlines () if len(m) > 3]

            for entry in entries:

                ### Step 1: extract the fields ###
                fields = [s.strip() for s in entry.split("|")]                                    
                meme_type = fields[0].lower().strip()
                top_text_raw = fields[1].lower().strip()
                bottom_text_raw = fields[2].lower().strip()

                ### Step 2: increment the count of examples of this meme type in self.meme_types ###
                if not meme_type in self.meme_types.keys ():
                    self.meme_types[meme_type] = 1
                else:
                    self.meme_types [meme_type] += 1

                ### Step 2: create the meme and add it to meme examples ###
                new_meme = Meme (self, meme_type, top_text_raw, bottom_text_raw)
                self.meme_examples.append (new_meme);

    def load_meme_examples_pickle (self):

        meme_examples_filename  = os.path.join (saved_data_directory, 'meme_examples.obj')
        meme_types_filename     = os.path.join (saved_data_directory, 'meme_types.obj')
        self.meme_examples      = pickle.load (open(meme_examples_filename, 'r'))
        self.meme_types         = pickle.load (open(meme_types_filename, 'r'))

    def save_meme_examples (self) :

        meme_examples_filename  = os.path.join (saved_data_directory, 'meme_examples.obj')
        meme_types_filename     = os.path.join (saved_data_directory, 'meme_types.obj')

        pickle.dump (self.meme_examples, open(meme_examples_filename, 'w'))
        pickle.dump (self.meme_types, open(meme_types_filename, 'w'))


    # Function: print_meme_examples_stats
    # -----------------------------------
    # prints out statistics on the loaded meme examples
    def print_meme_examples_stats (self):

        print_message ("Meme Example Stats:")
        for meme_type, count in self.meme_types.items ():
            print " ", meme_type, ": ", count
        print "\n"



    ########################################################################################################################
    ###############################[ --- Training/Loading/Saving the Classifier --- ]#######################################
    ########################################################################################################################
    # Function: get_training_data
    # ---------------------------
    # fills self.all_examples, self.training_examples and self.testing_examples with 
    # feature vector representations of all the memes we have access to
    def get_training_examples (self):

        ### Step 1: fill all_examples with feature representations of all memes ###
        self.all_examples = []
        for meme_example in self.meme_examples:
            self.all_examples.append ((meme_example.get_features(), meme_example.meme_type))

        ### Step 2: shuffle it up ###
        shuffle(self.all_examples)

        ### Step 3: divide it up for classification ###
        train_portion = 1.0
        num_training_examples = int(train_portion*len(self.all_examples))
        self.training_examples = self.all_examples [:num_training_examples]
        self.testing_examples = self.all_examples[num_training_examples:]

    # Function: train_classifier
    # --------------------------
    # trains the classifier based on self.training_examples 
    def train_classifier (self):

        algorithm = classify.MaxentClassifier.ALGORITHMS[0]
        self.classifier = classify.MaxentClassifier.train (self.training_examples, algorithm, trace=100, max_iter=3)


    # Function: (load|save)_classifier
    # --------------------------------
    # (un)pickle the trained classifier into (out of) a file
    def save_classifier (self, filename):

        full_filename = os.path.join (classifiers_directory, filename)
        pickle.dump (self.classifier, open(full_filename, 'w'))

    def load_classifier (self, filename):
        
        full_filename   = os.path.join (classifiers_directory, filename)
        self.classifier = pickle.load (open(full_filename, 'r')) 

   

    ########################################################################################################################
    ###############################[ --- Classification --- ]###############################################################
    ########################################################################################################################
    # Function: classify_features
    # ---------------------------
    # given a feature vector, this function will return a sorted probability distribution 
    # over the meme types
    def classify_features (self, features):
    
        p = self.classifier.prob_classify (features);
        prob_dist = {meme_type:p.prob(meme_type) for meme_type in p.samples()}
        sorted_prob_dist = sorted(prob_dist.iteritems(), key=operator.itemgetter(1), reverse=True);
        return sorted_prob_dist






    ########################################################################################################################
    ###############################[ --- Evaluating the Classifier --- ]####################################################
    ########################################################################################################################
    # Function: evalute_classifier_MRR
    # --------------------------------
    # this function will compute the MRR for the classifier 
    # returns the MRR
    def evaluate_classifier_MRR (self):

        total_rank = 0.0
        total_examples = 0.0

        for test_example in self.testing_examples:

            features = test_example[0]
            label = test_example[1]


            sorted_prob_dist = self.classify_features (features)
            ranked_meme_types = [m[0] for m in sorted_prob_dist]
            index = ranked_meme_types.index (label)
            total_rank += 1 / float(index + 1)
            total_examples += 1

        return (total_rank / total_examples)





    # # Function: unpack_to_defaultdict
    # # -------------------------------
    # # function to convert the pickled, non-defaultdict versions of tfidf and word_freq to defaultdicts.
    # def unpack_to_defaultdict(self, non_default):
    #     default = {meme_type:{c_name:defaultdict(lambda:0.0) for c_name in self.classifier_names} for meme_type in self.meme_types}
    #     for meme_type in non_default:
    #         for c_name in non_default[meme_type]:
    #             default[meme_type][c_name].update (non_default[meme_type][c_name])
    #     return default

    # # Function: load_vocab
    # # --------------------
    # # function to fill self.vocab; comes from the vocab being pickled in a file
    # def load_vocab_and_word_frequencies (self, f_vocab, f_word_freq):
    
    #     if f_vocab and f_word_freq:
    #         print "     retrieving from files: ", f_vocab, ", ", f_word_freq
    #         f = open(f_vocab, 'r')
    #         self.vocab = pickle.load(f)
            
    #         f = open (f_word_freq, 'r')
    #         word_freq_non_default = pickle.load (f)
    #         self.word_freq = self.unpack_to_defaultdict (word_freq_non_default)

    #     else:
    #         print "     computing..."
    #         self.get_vocab_and_word_frequencies ()
    #     return


    # # Function: maxent_load
    # # ---------------------
    # # will load parameters from a file, stores them in self.lambdas
    # def load_maxent (self, filenames):

    #     for c_name in self.classifier_names:
    #         if filenames[c_name] and filenames[c_name] != 'x':
    #             print "     loading ", c_name, " classifier from file: ", filenames[c_name]
    #             f = open(filenames[c_name], 'r')
    #             self.ngrams_classifier[c_name] = pickle.load(f)
    #             f.close ()
    #         elif filenames[c_name] == 'x':
    #             print "     not using ", c_name, " classifier"
    #         else:
    #             print "\n\n     training ", c_name, " classifier"
    #             self.train_ngrams_classifier (self.train_data, c_name)      
    
    #     return


    # # Function: load_ngrams_classifier
    # # --------------------------------
    # # loads parameters for ngram_classification from a file
    # def get_ngrams_classifier (self, train_data, filename):
        
    #     if filename:
    #         print "LOAD CLASSIFIER: (filename = " + filename + ")"
    #         f = open(filename, 'r')
    #         self.ngrams_classifier = pickle.load(f)
    #         f.close ()
        
    #     else:
    #         print "TRAIN CLASSIFIER:"
    #         self.train_ngrams_classifier (train_data);














    # Function: get_non_defaultdict
    # -----------------------------
    # this function will transform a both self.tfidf and self.word_frequencies into non-defaultdicts
    # in order that they can be pickled.
    def get_non_defaultdict (self, d):
        non_default = {}

        for k1 in d:
            non_default[k1] = {}
            for k2 in d[k1]:
                non_default[k1][k2] = {k:v for k, v in d[k1][k2].iteritems () if v > 0.0}

        return non_default


    # Function: Destructor
    # --------------------
    # will 'pickle' vocab, word_frequencies and tfidf if 'save' was set to true
    def __del__ (self):

        print "Shutting down..."

        # if self.save:
            # print "     pickling vocab..."
            # f = open ("vocab.obj", "w")
            # pickle.dump (self.vocab, f)

            # print "     pickling word_frequencies...."
            # f = open ("word_frequencies.obj", "w")
            # wf_non_default = self.get_non_defaultdict (self.word_frequencies)
            # pickle.dump(wf_non_default, f)


            # print "     pickling tfidf..."
            # f = open ("tfidf.obj", "w")
            # tfidf_non_default = self.get_non_defaultdict (self.tfidf)
            # pickle.dump(tfidf_non_default, f)

            # f.close() 
            # pass
        return






    ####################################################[ GETTING VOCAB, WORD FREQ, TFIDF ]##################################################

    #Function: get_word_frequencies
    #-----------------------
    # given a dict mapping meme types to a list of instances of them (i.e. self.memes), this function will fill 
    # self.word_frequencies (mapping meme-type to a dict that maps words to their frequencies) and fills maxent_training_data
    def get_vocab_and_word_frequencies (self):

        self.vocab = {c_name:set() for c_name in self.classifier_names}
        self.word_frequencies = {meme_type:{c_name:defaultdict(lambda: 0.0) for c_name in self.classifier_names} for meme_type in self.meme_types}

        for meme_type, instances in self.memes.iteritems ():
            for instance in instances:
                for c_name in self.classifier_names:

                    for word in instance[c_name]:
                        self.word_frequencies[meme_type][c_name][word] += 1

                    self.vocab[c_name].add (word)

        return






    ####################################################[ FEATURE EXTRACTION ]################################

    # Function: get_tfidf_features (self, sentence):
    # ----------------------------------------------
    # this function will return a dict mapping meme_type[c_i] to the product of all tf_idf(w_j, c_i), 
    # where w_j is jth word in the meme instance. 
    def extract_tfidf_features (self, sentence, c_name):
        
        tfidf_features = {}
        for meme_type in self.meme_types:
            product = 1.0
            for word in sentence:
                if (self.tfidf[meme_type][c_name][word] > 0):
                    product *= self.tfidf[meme_type][c_name][word]
            tfidf_features[meme_type] = product
        return tfidf_features











    ####################################################[ TRAINING ]##################################################
    
    # Function: get_labeled_data
    # ------------------------------
    # fills self.labelled_examples with the appropriate data.
    def get_labeled_data (self):

        # self.labeled_data = {c_name:[] for c_name in self.classifier_names}
        self.labeled_data = []

        for meme_type, instances in self.memes.iteritems ():
            for instance in instances:

                for c_name in self.classifier_names:

                    features[c_name] = self.extract_features (instance[c_name], c_name)
                
                self.labeled_data.append ((features, meme_type));
        
        shuffle (self.labeled_data)
        return



    # Function: get_labeled_data_tfidf
    # --------------------------------
    # fills self.labelled_examples with tfidf scores for each class.
    def get_labeled_data_tfidf (self):
        self.labeled_data = []
        features = {}

        for meme_type, instances in self.memes.iteritems ():
            for instance in instances:

                for c_name in self.classifier_names:

                    features[c_name] = self.extract_tfidf_features(instance[c_name], c_name);

                self.labeled_data.append ((features, meme_type));


        shuffle (self.labeled_data);
        return









    # Function: train_ngrams_classifier
    # ----------------------
    # will train the ngrams classifier then immediately save it
    # c_name = the name of the classifier, i.e. top/bottom/all
    def train_ngrams_classifier (self, train_data):
    
        self.ngrams_classifier = MaxentClassifier.train (train_data, trace=100, max_iter=5)
        
        if self.save:
            save_name = "ngrams_classifier.obj"
            f = open (save_name, "w")
            pickle.dump (self.ngrams_classifier, f)
            print "     ##### saved at: ", save_name, " #####"
            f.close ()
        return








    ####################################################[ CLASSIFICATION ]##################################################

    # Function: merge_distributions
    # -----------------------------
    # given a dict prob_dists s.t. prob_dists[c_name][meme_type] = prob of meme_type under c_name, returns
    # a dict prob_dist s.t. prob_dist[meme_type] = prob of that meme overall
    def merge_distributions (self, prob_dists):
        
        #print prob_dists

        output_prob_dist = {}       
        for meme_type in self.meme_types:
            prob_product = 1.0
            for c_name in self.classifier_names:
                prob_product *= prob_dists[c_name][meme_type]

            output_prob_dist[meme_type] = prob_product

        return output_prob_dist



    # Function: classify_meme_raw
    # ------------------------------
    # given a dict representing a meme (keys = top/bottom/all), this will create a meme object out of it 
    # and apply self.classify_meme to it.
    def classify_meme_raw (self, meme_example):

        print "--- classify_meme_raw ---"
        top_text_raw = meme_example['top'].lower()
        print "Checkpoint 0.0"
        bottom_text_raw = meme_example['bottom'].lower()
        print "Checkpoint 0.1"
        new_meme = Meme (self, None, top_text_raw, bottom_text_raw);
        print "Checkpoint 0.2"
        print "--- entering classify_meme... ---"
        sorted_prob_dist = self.classify_meme (new_meme)
        print "--- finished classify_meme ---"
        return sorted_prob_dist


    # Function: classify_meme
    # ------------------------------
    # given a Meme object, this function will return a probability distribution over the meme_types,
    # sorted in decreasing order.
    # ----
    # NOTE: currently based only on ngram-features.
    def classify_meme (self, meme_example):

        print "--- classify_meme ---"

        ngram_features = meme_example.ngram_features
        p = self.ngrams_classifier.prob_classify(ngram_features)
        prob_dist = {meme_type:p.prob(meme_type) for meme_type in p.samples()}
        sorted_prob_dist = sorted(prob_dist.iteritems(), key=operator.itemgetter(1), reverse=True);

        print "--- end ---"
        return sorted_prob_dist




    # Function: generate_all_possible_partitions
    # -------------------------------------
    # given a raw sentence, this function will generate all possible 'partitions' that the sentence
    # could have and will store/return them in a list of Meme objects
    def generate_all_possible_partitions (self, raw_sentence):


        sentences = sent_tokenize (raw_sentence)
        tokenized_sentences = [word_tokenize (s) for s in sentences]
        token_list = []

        possible_memes = []
        
        for s in tokenized_sentences:
            for word in s:
                token_list.append(word)

        for i in range(len(token_list) + 1):
            top_text_tokenized = token_list[0:i]
            bottom_text_tokenized = token_list[i:]

            top_text_raw = ' '.join(top_text_tokenized)
            bottom_text_raw = ' '.join(bottom_text_tokenized)

            new_meme = Meme (self, None, top_text_raw, bottom_text_raw)
            possible_memes.append (new_meme)

        return possible_memes





    # Function: classify_sentence
    # ---------------------------
    # given the raw text of a sentence, this function will return a probability distribution over the meme_types,
    # sorted in decreasing order
    # ---------
    # note - how do we get the partition that it scored well on??
    def classify_sentence (self, raw_sentence):

        ### Step 1: generate all possible meme versions of the sentence, or  memes ###
        possible_partitions = self.generate_all_possible_partitions (raw_sentence)

        ### Step 2: get prob_dist for each possible partition, store (partition, sorted_prob_dist) tuples in meme_array  ###
        meme_array = []
        for partition in possible_partitions:

            p = self.ngrams_classifier.prob_classify(partition.ngram_features)
            prob_dist = {meme_type:p.prob(meme_type) for meme_type in p.samples()}
            sorted_prob_dist = sorted(prob_dist.iteritems(), key=operator.itemgetter(1), reverse=True);


            meme_array.append ((partition, sorted_prob_dist))


        # print "##### CLASSIFY SENTENCE #####"
        max_probs = []
        for meme_type in self.meme_types:

            max_prob = 0
            max_partition = meme_array[0][0]

            for element in meme_array:
                current_partition = element[0]

                current_dist = element[1]
                names = [d[0] for d in current_dist] #contains just all the names

                if meme_type in names:
                    index = names.index(meme_type)
                    current_prob = current_dist[index][1]

                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_partition = current_partition

            max_probs.append ((meme_type, max_prob, max_partition))

        ### go from max_probs - > sorted max_probs ###
        sorted_prob_dist = sorted(max_probs, key=operator.itemgetter(1), reverse=True)

        return sorted_prob_dist


    # Function: get_tfidf_distribution
    # --------------------------------
    # given a meme instance, returns a distribution over the classes based on tfidf scores
    def get_tfidf_distribution (self, instance):
        dist = {}

        for key in self.classifier_names:
            dist[key] = self.extract_tfidf_features();


        return dist


    # Function: get_ngrams_distribution
    # ---------------------------------
    # given a meme instance, returns a distribution over the classes based on ngrams
    # returns dist[key], s.t. dist[bottom] is a ranking of all of the meme types as (name, prob) tuples
    def get_ngrams_distribution (self, instance):
        dist = {}
        features[key] = self.extract_ngram_features ();
        dist[key] = self.classifier[key].prob_classify (features[key])

        return dist




    # Function: maxent_classify_sentence
    # -------------------------
    # given a sentence, this function will return a dict mapping classifier names (bottom/top/all)
    # to probability distributions over all the classes (meme-types)
    def maxent_classify_sentence (self, instance):
        dist = {}
        features = {}

        for key in self.classifier_names:
            features[key] = self.extract_features ()
            dist[key] = self.classifier[key].prob_classify (features[key]) 

        return dist


    # Function: maxent_classify_raw
    # -----------------------------
    # given a raw sentence, this will tokenize it then classify it.
    def maxent_classify_raw (self, sentence_raw):
        
        sentence = wordpunct_tokenize(sentence_raw)
        return self.maxent_classify(sentence)













    ####################################################[ GETTING LABELLED DATA FOR TEST/TRAIN ]##################################################

    # Function: get_ngrams_labeled_data
    # ---------------------------------
    # fills self.ngrams_labeled_data with labeled feature vectors, one for each meme
    def get_ngrams_labeled_data (self):

        self.ngrams_labeled_data = []

        for meme_example in self.meme_examples:
            labeled_data = (meme_example.ngram_features, meme_example.meme_type);
            self.ngrams_labeled_data.append (labeled_data)

        shuffle (self.ngrams_labeled_data)

    # Function: partition_ngrams_labeled_data
    # ------------------------
    # this function will partition the data into test/training sets
    # the cv_val passed in is what fraction of the data is in the training set.
    # note: only call after get_labeled_data!
    def partition_ngrams_labeled_data (self, cv_val):

        n = self.num_of_meme_examples * cv_val
        n = int(n)

        self.ngrams_train_data = self.ngrams_labeled_data[:n]
        self.ngrams_test_data = self.ngrams_labeled_data[n:]









    ####################################################[ EVALUATION ]##################################################
    # Function: maxent_evaluate
    # -------------------------
    # this function will run 5-fold CV and do Mean Reciprocal Rank to tell you what your score is.
    def maxent_evaluate_old (self):

        score_total = {c_name:0.0 for c_name in self.classifier_names}
        score_total['merged'] = 0.0


        for instance in self.test_data:
            features = instance[0]
            correct_label = instance[1]

            score = {}

            dist_sorted = {}
            for c_name in self.classifier_names:
                p = self.classifier[c_name].prob_classify (features[c_name])
                samples = p.samples ()

                dist = [(s, p.prob(s)) for s in samples]
                dist_sorted[c_name] = sorted(dist, key=lambda tup: tup[1], reverse=True)

                names = [d[0] for d in dist_sorted[c_name]]

                score_total[c_name] += 1.0/float(names.index(correct_label) + 1)




        print "total scores:"
        for c_name in self.classifier_names:
            print "     ", c_name, ": "
            print "     total score: ", score_total[c_name]
            print "     MRR: ", score_total[c_name] / float(len(self.test_data))
        print "     merged: "
        print "     total score: ", score_total['merged']
        print "     MRR: ", score_total['merged'] / float(len(self.test_data))


        return

    # Function: maxent_evaluate
    # -------------------------
    # this function will run 5-fold CV and do Mean Reciprocal Rank to tell you what your score is.
    # this function uses only the maxent trained on unigrams
    def maxent_evaluate (self):

        score_total = 0.0

        for instance in self.test_data:
            features = instance[0]
            correct_label = instance[1]

            score = {}

            prob_dist_by_c_name = {}
            for c_name in self.classifier_names:
                p = self.classifier[c_name].prob_classify (features[c_name])
                prob_dist_by_c_name[c_name] = {meme_type:p.prob(meme_type) for meme_type in p.samples() }


            prob_dist = self.merge_distributions(prob_dist_by_c_name)
            merged_dist = sorted(prob_dist.iteritems(), key=operator.itemgetter(1), reverse=True);

            names = [d[0] for d in merged_dist]
            score_total += 1.0/float(names.index(correct_label) + 1)



        print "total scores:"
        print "     merged: "
        print "     total score: ", score_total
        print "     MRR: ", score_total / float(len(self.test_data))


        return


    # Function: evaluate_ngrams_classifier
    # ------------------------------------
    # given test data, this function will return the MRR of the ngram_classifier's performance on it.
    def evaluate_ngrams_classifier (self, test_data):

        score_total = 0.0

        for example in test_data:

            ### Step 1: get the correct label and features ###
            correct_label = example[1]
            features = example[0]

            ### Step 2: classify it ###
            p = self.ngrams_classifier.prob_classify (features);

            ### Step 3: get a sorted probability distribution out of the return value ###
            prob_dist = {meme_type:p.prob(meme_type) for meme_type in p.samples()}
            sorted_prob_dist = sorted(prob_dist.iteritems(), key=operator.itemgetter(1), reverse=True);


            ### Step 4: figure out how well we did ###
            names = [d[0] for d in sorted_prob_dist]
            score_total += 1.0/float(names.index(correct_label) + 1)


        print "----- NGRAM_CLASSIFIER EVALUATION RESULTS -----"
        print "     total score: ", score_total
        print "     number of examples: ", len(test_data)
        print "     -> MRR: ", score_total / float(len(test_data))













####################################################[ PROGRAM OPERATION ]##################################################


if __name__ == "__main__":

    filenames = defaultdict(lambda: None)

    filenames['meme_examples'] = [
        "../old_data/Foul-Bachelor-Frog data.txt",
        "../old_data/Futurama-Fry data.txt",
        "../old_data/Good-Guy-Greg- data.txt",
        "../old_data/Insanity-Wolf data.txt",
        "../old_data/Philosoraptor data.txt",
        "../old_data/Scumbag-Steve data.txt",
        "../old_data/Socially-Awkward-Penguin data.txt",
        "../old_data/Success-Kid data.txt",
        "../old_data/Paranoid-Parrot data.txt",
        "../old_data/Annoying-Facebook-Girl data.txt",
        "../old_data/First-World-Problems data.txt",
        "../old_data/Forever-Alone data.txt",
        "../old_data/The-Most-Interesting-Man-In-The-World data.txt"
    ]

    filenames['meme_types'] = "../data/meme_types.txt"
    # filenames['sentiment'] = '../data/inquirerbasic.csv'



    # if 'load_word_stats' in sys.argv:
        # filenames['vocab']              = 'vocab.obj'
        # filenames['word frequencies']   = 'word_frequencies.obj'
        # filenames['tfidf']              = 'tfidf.obj'

    # elif 'load_all' in sys.argv:
        
    if 'load' in sys.argv:
        # filenames['vocab']              = 'vocab.obj'
        # filenames['word frequencies']   = 'word_frequencies.obj'
        # filenames['tfidf']              = 'tfidf.obj'

        # filenames['all'] = 'classifier_all.obj'
        # filenames['top'] = 'classifier_top.obj'
        # filenames['bottom'] = 'classifier_bottom.obj'

        filenames['ngrams_classifier'] = 'ngrams_classifier.obj'


    if 'save' in sys.argv:
        save = True
    else:
        save = False

    automeme = Automeme (filenames, save, mode='dev_mode')

        
