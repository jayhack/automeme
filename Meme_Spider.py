# ---------------------------------------------------------- #
# Class: Meme_Spider.py
# ---------------------
# an abstract class used for scraping memes from sites.
#
# ---------------------------------------------------------- #

#--- Standard ---
import os
import sys 
import pickle
import time
import csv
from collections import defaultdict

#--- Scrapy ---
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
from meme_scraper.items import Meme_Item



# Class: DiyLolSpider
# -----------------
# spider to crawl through quickmeme and scrape text from advice animals.
# currently only scrapes meme types that are hard-coded in.
class MemeSpider (BaseSpider):

    #--- Scraper Identity ---
    name = None                         #fill this in
    allowed_domains = []
    start_urls = []

    #--- on the list of advice animals we will be scraping... ---
    current_meme_name   = ''
    data_directory      = os.path.join(os.getcwd(), 'data/' + name)
    meme_types_filename = os.path.join (data_directory, 'meme_types.txt')
    captions_filename = ''

    #--- Data Parameters ---
    current_page_index = defaultdict(lambda: 1)
    max_meme_instances = 10000

    #--- Data ---
    meme_types      = []                           #list of meme_types         
    meme_counts     = defaultdict(lambda: 0)      #dict mapping meme_type -> number of instances gathered


    ########################################################################################################################
    ###################[ --- Constructor/Initialization/Destructor --- ]###################################################################
    ########################################################################################################################

    # Function: check_name
    # --------------------
    # ensures this scraper has an actual name
    def check_name (self):
        if not self.name:
            print_error ("Initialize", "This spider does not have a name!")


    # Function: get_filenames
    # -----------------------
    # sets up the filenames to read from/write to
    def set_filenames (self):

        self.data_directory         = os.path.join (os.getcwd(), 'data/' + self.name)
        self.meme_types_filename    = os.path.join (data_directory, 'meme_types.txt')


    # Function: get_meme_types
    # ------------------------
    # loads the names of all memes that we are going to scrape
    def get_meme_types (self, print_meme_types=False):

        ### Step 1: set the meme types appropriately ###
        self.meme_types = [w.strip() for w in open(self.meme_types_filename, 'r').readlines()]        
        
        ### Step 2: print out the meme types if requested ###
        if print_meme_types:
            print "##### Meme Types to be Scraped: #####"
            for meme_type in self.meme_types:
                print " " + meme_type   


    # Function: initialize
    # --------------------
    # call this function in order to initialize an instantiated 
    # version of MemeSpider
    def initialize (self):

        ### Step 1: make sure that the spider has a name ###
        self.check_name ()

        ### Step 2: set up the filenames to read from/write to ###
        self.set_filenames ()

        ### Step 3: get self.meme_types ###
        self.get_meme_types (print_meme_types=True)


    # Function: start_requests
    # ------------------------
    # sets up initial requests: one for each meme_type
    def start_requests (self):
        
        requests = []
        for meme_type in self.meme_types:
            requests.append ( Request (url=self.get_meme_page_url(meme_type), meta={'meme_type':meme_type}) )
        return requests








    ########################################################################################################################
    ###################[ --- URL Transformation Utilities --- ]#############################################################
    ########################################################################################################################

    # Functions: get_meme_page_url
    # ----------------------------
    # given a meme name, this returns the url of its first page
    def get_meme_page_url (self, meme_name):

        pass

    # Function: get_next_page_url
    # ---------------------------
    # given an htmlxpathselector, this will return an xpath to 
    # the next page. returns None if there are no more pages
    def get_next_page_url (self, meme_name):

        pass











    ########################################################################################################################
    ###################[ --- Getting Memes --- ]############################################################################
    ########################################################################################################################

    # Function: parse
    # ---------------
    # gets called on the main pages that have thumbnails
    def parse (self, response):

        pass


    # Function: parse_meme_page
    # -------------------------
    # parses a page dedicated to a specific meme
    def parse_meme_page (self, response):
        
        pass










