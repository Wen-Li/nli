import os
from nltk.parse.stanford import StanfordDependencyParser

path_prefix = "/Users/w2li/Documents/nli/data/essays/train"
parser_path = "/usr/local/Cellar/stanford-parser/3.6.0/bin/lexparser.sh"

in_path = path_prefix + "/original"
out_path = path_prefix + "/parsed"

os.environ['STANFORD_PARSER'] = '/usr/local/Cellar/stanford-parser/3.6.0/libexec'
os.environ['STANFORD_MODELS'] = '/usr/local/Cellar/stanford-parser/3.6.0/libexec'

# dep_parser = StanfordDependencyParser()
dep_parser = StanfordDependencyParser(model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

for dep_graphs in dep_parser.raw_parse_sents(open("../data/essays/train/tokenized/00001.txt").readlines()):
    for parse in dep_graphs:
        for i in parse.triples():
                print(i)
    print()
# for subdir, dirs, files in os.walk(in_path):
#     for filename in files:
#         in_filename = os.path.join(subdir, filename)
#         out_filename = os.path.join(out_path, filename)
#         print("Parsing {} ...".format(in_filename))
#         os.system("{} {} > {}".format(parser_path, in_filename, out_filename))
