import os
from nltk.stem import WordNetLemmatizer

############ TRAIN, DEV, OR TEST
path_prefix = "/Users/w2li/Documents/nli/data/essays/dev"

############ from TOKENIZED to STEMMED
in_path = path_prefix + "/tokenized"
out_path = path_prefix + "/lemmatized"

wnl = WordNetLemmatizer()

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        in_filename = os.path.join(subdir, filename)
        out_filename = os.path.join(out_path, filename)
        print("Lemmatizing {} ...".format(in_filename))
        try:
            f = open(in_filename)
            f_writer = open(out_filename, "w")
            lemmas = []
            for word in f.read().split():
                lemmas.append(wnl.lemmatize(word))
            f_writer.write(" ".join(lemmas))
            f.close()
            f_writer.close()
        except Exception as e:
            print(e)
