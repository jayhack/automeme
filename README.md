-----------------------------------------------------

Automeme
========
Language modeling and classification for document 
classes defined by 'Memes' (AdviceAnimals)

by Jay Hack & Sam Beder, Fall 2013

-----------------------------------------------------

1: Description
==============

TODO: fill this out

Given a sentence

1.1: File Structure
-------------------

all source files are contained in the base directory

in data:

• data/memes contains a list of json objects representing
memes
• data/

2.1: Time Performance
---------------------
(all of this on MBPro)

• Pandas load: takes < 5 seconds

• Tokenizer: takes < 20 seconds

• BOW representation: takes < 10 seconds

• BOW -> vocab mat: < 15 seconds

• sklearn Logistic Regression fit(X, y): about 4 mins!

2.2: Classification Performance
-------------------------------

• Not too great right now... need better data

2: Setup
========

2.1: Libraries
--------------

Automeme uses the following libraries:

• nltk

• 

