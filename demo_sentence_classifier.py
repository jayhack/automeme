#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    demo_meme_classifier.py
    -----------------------

    A demonstration of meme-classification
    
"""
import sys
import os
from SentimentAnalysis import SentimentAnalysis
import pickle
import itertools
import json
from operator import itemgetter
from collections import defaultdict
from flask import Flask, request, render_template, jsonify, redirect

import urllib
import urllib2
import lxml
import lxml.html
from lxml import etree



#############################################################################################################################
##############################[ --- SENTIMENT ANALYSIS FILENAMES AND SETUP ---]##############################################
#############################################################################################################################

filenames = defaultdict(lambda: None)
filenames['ngrams_classifier'] = "ngrams_classifier.obj"
filenames['meme_types'] = "meme_types.txt"

# filenames['sentiment'] = '../data/inquirerbasic.csv'
# filenames['vocab']              = '../process_data/vocab.obj'
# filenames['word frequencies']   = '../process_data/word_frequencies.obj'
# filenames['tfidf']              = '../process_data/tfidf.obj'
# filenames['all'] = '../process_data/classifier_all.obj'
# filenames['top'] = '../process_data/classifier_top.obj'
# filenames['bottom'] = '../process_data/classifier_bottom.obj'


# --------- load the sentiment analysis classifier -----------
save = False
sa = SentimentAnalysis (filenames, save, mode='use_mode')









#############################################################################################################################
##############################[ --- URLS FOR GETTING IMAGES, GENERATING MEMES ---]###########################################
#############################################################################################################################
image_locations = {
    'Annoying Facebook Girl': 'http://i.imgflip.com/1bhi.jpg',
    'First World Problems': 'http://i.imgflip.com/1bhf.jpg',
    'Forever Alone': 'http://i.imgflip.com/1bh4.jpg',
    'Foul Bachelor Frog': 'http://i.imgflip.com/1bgv.jpg',
    'Futurama Fry': 'http://i.imgflip.com/1bgw.jpg',
    'Good Guy Greg': 'http://i.imgflip.com/1bgx.jpg',
    'Insanity Wolf': 'http://i.imgflip.com/1bgu.jpg',
    'Paranoid Parrot': 'http://i.imgflip.com/1bi4.jpg',
    'Philosoraptor': 'http://i.imgflip.com/1bgs.jpg',
    'Scumbag Steve': 'http://i.imgflip.com/1bgy.jpg',
    'Socially Awkward Penguin': 'http://i.imgflip.com/1bh0.jpg',
    'Success Kid': 'http://i.imgflip.com/1bhk.jpg',
    'The Most Interesting Man In The World': 'http://i.imgflip.com/1bh8.jpg',
    'Socially Awesome Penguin': 'http://i.imgflip.com/1bgz.jpg'
    
}
request_locations = {
    'Success Kid':'http://diylol.com/meme-generator/success-kid/memes',
    'Annoying Facebook Girl': 'http://diylol.com/meme-generator/annoying-facebook-girl/memes',
    'First World Problems': 'http://diylol.com/meme-generator/first-world-problems/memes',
    'Forever Alone': 'http://diylol.com/meme-generator/forever-alone/memes',
    'Foul Bachelor Frog': 'http://diylol.com/meme-generator/foul-bachelor-frog/memes',
    'Futurama Fry': 'http://diylol.com/meme-generator/futurama-fry/memes',
    'Good Guy Greg': 'http://diylol.com/meme-generator/good-guy-greg/memes',
    'Insanity Wolf': 'http://diylol.com/meme-generator/insanity-wolf/memes',
    'Paranoid Parrot': 'http://diylol.com/meme-generator/paranoid-parrot/memes',
    'Philosoraptor': 'http://diylol.com/meme-generator/philosoraptor/memes',
    'Scumbag Steve': 'http://diylol.com/meme-generator/scumbag-steve/memes',
    'Socially Awkward Penguin': 'http://diylol.com/meme-generator/socially-awkward-penguin/memes',
    'The Most Interesting Man In The World': 'http://diylol.com/meme-generator/The-Most-Interesting-Man-In-The-World/memes',
    'Socially Awesome Penguin': 'http://diylol.com/meme-generator/Socially-Awesome-Penguin/memes'
} 








#############################################################################################################################
##############################[ --- FLASK APP FUNCTIONS ---]#################################################################
#############################################################################################################################



app = Flask(__name__)

# Function: index
# ---------------
# this merely returns the baseline site
@app.route('/')
def index():
    return render_template('sentence_classifier.html')




# Function: classify_meme
# ----------------------------
# given the top/bottom text for a meme, this function will return a list of the memes that
# can fit it, ranked by how probable they are (most probable first) 
@app.route("/_classify_meme")
def classsify_meme ():

    print "---> STATUS: Classifying meme"

    top_text = request.args.get('top_text', 0, type=str)

    print top_text


    sorted_prob_dist = sa.classify_sentence (top_text)
    print "### Classification Results: ###"
    print sorted_prob_dist

    r = [s[0] + (" - %f" % s[1]) for s in sorted_prob_dist]
    urls = [[s[0], image_locations[s[0]]] for s in sorted_prob_dist]
    print "checkpoint 2"
    packed_rankings = json.dumps (urls)

    print "---> STATUS: Returning distribution over memes"

    return jsonify(result=packed_rankings)




# Function: generate_meme
# -----------------------
# given the top text, bottom text and the index of the clicked image, this will generate 
# the meme appropriately and send back an URL that is hosting the meme.
@app.route("/_generate_meme")
def generate_meme ():


    print "---> STATUS: Generating Meme"

    top_text = request.args.get('top_text', 0, type=str)
    bottom_text = request.args.get('bottom_text', 0, type=str)  
    meme_type = request.args.get('meme_name', 0, type=str)


    if top_text == '':
        top_text = 'you need to enter text here...'
    if bottom_text == '':
        top_text = 'you need to enter text here...'


    post_data_dict = {  
                        'post_line1':top_text, 
                        'post[line1]':top_text, 
                        'post[line2]':bottom_text, 
                        'post_line2':bottom_text
    }
    post_data_encoded = urllib.urlencode(post_data_dict)


    meme_generation_url = request_locations[meme_type];

    request_object = urllib2.Request(meme_generation_url, post_data_encoded)
    response = urllib2.urlopen(request_object)
    html_string = response.read()
    html = etree.HTML(html_string);
    result = html.xpath('//div[@class="img"]/a/img/@src')


    image_url = result[0]
    packed_image_url = json.dumps(image_url)


    return jsonify(result=packed_image_url);




if __name__ == '__main__':

    app.debug = True
    app.run()
