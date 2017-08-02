import os

########### INPUT & OUTPUT PATHS
path_prefix = "/Users/w2li/Documents/nli/data/essays/test"
in_path = path_prefix + "/tokenized"
out_path = path_prefix + "/tagged"

tagger_path = "/Users/w2li/Documents/stanford-postagger-2016-10-31/stanford-postagger.jar"
model_path = "/Users/w2li/Documents/stanford-postagger-2016-10-31/models/english-left3words-distsim.tagger"

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        ############ start point
        if int(filename[:4]) == 2136:
            in_filename = os.path.join(subdir, filename)
            out_filename = os.path.join(out_path, filename)
            print("Parsing {} ...".format(in_filename))
            try:
                os.system("java -cp {} edu.stanford.nlp.tagger.maxent.MaxentTagger -model {} -textFile {} -outputFormat slashTags -sentenceDelimiter newline -tokenize false > {}".format(
                    tagger_path, model_path, in_filename, out_filename
                ))
            except Exception as e:
                print(e)
