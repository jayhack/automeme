#!/usr/bin/python
# ---------------------------------------------------------- #
# File: Preprocess.py
# -----------------
# contains everything pertaining to preprocessing memes
# (i.e. filtering, etc)
# ---------------------------------------------------------- #
#--- Standard ---
import json
import pickle

#--- Langid ---
import langid

#--- My files ---
from Meme import Meme 


# Function: length_is_ok
# ----------------------
# given a meme, returns true if the meme's length is sufficient
# total length currently set at 5, top at 1, bottom at 1
def length_is_ok (meme):
    
    top_length = len([w for w in meme.top_text if w.isalpha ()])
    bottom_length = len([w for w in meme.bottom_text if w.isalpha()])
    total_length = top_length + bottom_length

    if total_length >= 5 and top_length >= 1 and bottom_length >= 1:
        return True
    else:
        return False


# Function: is_english
# --------------------
# uses langid module to ensure memes are in english
# returns true if they are, in fact, english
def is_english (meme):    

    return (langid.classsify (meme.all_text)[0] == 'en')


# Function: filter_memes
# ----------------------
# iterates through all memes, filters out ones that are:
# - not in english
# - too short
def filter_memes (memes):

    ### --- iterate through all memes, remove ones that are too short/not english --- ###
    remove_indeces = []
    for meme_type, memes_list in memes.iteritems ():
	    for index, meme in enumerate(memes_list):

	        if not (length_is_ok (meme) and is_english(meme)):
	            remove_indeces.append (index)
	            print "=== removed ==="
	            print meme
	            print "\n"

	    meme_list = [meme for index, meme in enumerate(memes) if not index in remove_indeces]
	    return memes

           
