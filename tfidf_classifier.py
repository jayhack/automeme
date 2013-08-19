# Class: Tfidf_classifier
# -----------------------
# takes care of all things tf.idf related, including classification
class Tfidf_classifier:

    parent = None;

    def __init__ (self, parent, filename):

        self.parent = parent;

        if filename:
            print "     retrieving from file: ", filename
            f = open(filename, 'r')
            tfidf_non_default = pickle.load(f)
            parent.tfidf = parent.unpack_to_defaultdict (tfidf_non_default)
       # elif False:   
           # print "     computing..."
           # self.compute_tfidf ()
        else:
            print "     Error: enter a viable option for tf.idf calculation."


    # tf (term frequency): number of times that word occurs in meme_type under c_name
    def compute_tf(self, meme_type, c_name, word):
        tf = self.word_frequencies[meme_type][c_name][word]
        if tf > 0.0:
            return 1.0 + math.log10(tf)
        else:
            return 0.0

    # df (document frequency): number of meme instances total that contain word under c_name
    def compute_df (self, c_name, word):

        df = 0.0
        for meme_type, instances in self.memes.iteritems():
            for instance in instances:
                if word in instance[c_name]:
                    df += 1
        return df

    # idf (inverse document frequency): log10(N/df) where N = total # of meme instances
    def compute_idf(self, c_name, word):
        df = self.compute_df (c_name, word)
        idf = math.log10(self.num_of_meme_instances/df)
        return idf

    # Functions: compute_tfidf
    # ------------------------
    # these functions will compute the tf-idf scores for all words across all the meme-types.
    def compute_tfidf(self):

        parent.tfidf = {meme_type:{c_name:defaultdict(lambda: 0.0) for c_name in parent.classifier_names} for meme_type in parent.meme_types}
        
        for meme_type in parent.memes:
            for c_name in parent.classifier_names:

                for word in parent.vocab[c_name]:
                    if parent.word_frequencies[meme_type][c_name][word] > 0.0:
                        parent.tfidf[meme_type][c_name][word] = self.compute_tf(meme_type, c_name, word) * self.compute_idf(c_name, word)

        return
