#!/usr/bin/python
import os
import sys
import json


# Function: parse_file
# --------------------
# returns a list of meme dicts from a string rep
# of a file
def parse_file (filename):

	memes = []
	lines = open(filename, 'r').readlines ()
	for line in lines:
		if not len(line) > 5:
			continue
		splits = line.split ('|')
		meme_type = splits[0].strip ().lower ()
		top_text = splits[1].strip ().lower ()
		bottom_text = splits[2].strip ().lower ()

		meme_type = meme_type.replace (' ', '-')

		memes.append ({'meme_type':meme_type, 'top_text':top_text, 'bottom_text':bottom_text})

	return memes



if __name__ == "__main__":

	filenames = [f for f in os.listdir (os.getcwd ()) if f[-3:] == 'txt']

	for filename in filenames:
		memes = parse_file (filename)

		meme_type = memes[0]['meme_type']

		output_filename = meme_type + '_instances.json'
		output_string = json.dumps (memes)
		outfile = open(output_filename, 'w')
		outfile.write (output_string)





