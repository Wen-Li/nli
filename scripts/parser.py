import os
from nltk.parse.stanford import StanfordDependencyParser

path_prefix = "/Users/w2li/Documents/nli/data/essays/test"
parser_path = "/usr/local/Cellar/stanford-parser/3.6.0/bin/lexparser.sh"

in_path = path_prefix + "/tokenized"
out_path = path_prefix + "/parsed"

os.environ['STANFORD_PARSER'] = '/usr/local/Cellar/stanford-parser/3.6.0/libexec'
os.environ['STANFORD_MODELS'] = '/usr/local/Cellar/stanford-parser/3.6.0/libexec'

dep_parser = StanfordDependencyParser(model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        ############ start point and end point
        # if int(filename[:5]) in [335, 336, 337, 338, 339, 340, 341, 342, 373, 374]:
        in_filename = os.path.join(subdir, filename)
        out_filename = os.path.join(out_path, filename)
        print("Parsing {} ...".format(in_filename))
        try:
            f = open(in_filename)
            f_writer = open(out_filename, "w")

            for dep_graphs in dep_parser.parse_doc(f.read()):
                for parse in dep_graphs:
                    for i in parse.triples():
                        if len(i) == 3:
                            f_writer.write("_".join([i[0][0], i[1], i[2][0]]))
                            f_writer.write("\n")
                            f_writer.write("_".join([i[0][1], i[1], i[2][0]]))
                            f_writer.write("\n")
                            f_writer.write("_".join([i[0][0], i[1], i[2][1]]))
                            f_writer.write("\n")
                            f_writer.write("_".join([i[0][1], i[1], i[2][1]]))
                            f_writer.write("\n")
            f.close()
            f_writer.close()
        except Exception as e:
            print(e)
