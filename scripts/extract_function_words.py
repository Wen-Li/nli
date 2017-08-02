import os

############ TRAIN, DEV, OR TEST
path_prefix = "/Users/w2li/Documents/nli/data/essays/test"

############ from TOKENIZED to STEMMED
in_path = path_prefix + "/tagged"
out_path = path_prefix + "/function_words"

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        if int(filename[:4]) == 2136:
            in_filename = os.path.join(subdir, filename)
            out_filename = os.path.join(out_path, filename)
            print("Extracting function words from {} ...".format(in_filename))
            try:
                f = open(in_filename)
                f_writer = open(out_filename, "w")
                function_words = []
                for word_pos in f.read().split():
                    word, pos = word_pos.split("_")
                    # CC, CD, DT, EX, IN, MD, PDT, POS, PRP, PRP$, TO, WDT, WP, WP$, WRB
                    if pos[0] in "CDEIMPTW":
                        function_words.append(word.lower())
                f_writer.write(" ".join(function_words))
                f.close()
                f_writer.close()
            except Exception as e:
                print(e)
