import os
from nltk.stem import PorterStemmer

############ TRAIN, DEV, OR TEST
path_prefix = "/Users/w2li/Documents/nli/data/essays/test"

############ from TOKENIZED to STEMMED
in_path = path_prefix + "/tokenized"
out_path = path_prefix + "/stemmed"

ps = PorterStemmer()

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        in_filename = os.path.join(subdir, filename)
        out_filename = os.path.join(out_path, filename)
        print("Stemming {} ...".format(in_filename))
        try:
            f = open(in_filename)
            f_writer = open(out_filename, "w")
            stems = []
            for word in f.read().split():
                stems.append(ps.stem(word).lower())
            f_writer.write(" ".join(stems))
            f.close()
            f_writer.close()
        except Exception as e:
            print(e)
