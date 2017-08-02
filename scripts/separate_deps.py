import os
import numpy as np

############ TRAIN, DEV, OR TEST
path_prefix = "/Users/w2li/Documents/nli/data/essays/train_dev"

############ from TAGGED to DEP0, DEP1, DEP2
in_path = path_prefix + "/parsed"
out_path_0 = path_prefix + "/dep0"
out_path_1 = path_prefix + "/dep1"
out_path_2 = path_prefix + "/dep2"

for p in [out_path_0, out_path_1, out_path_2]:
    if not os.path.exists(p):
        os.makedirs(p)

for subdir, dirs, files in os.walk(in_path):
    for filename in files:
        in_filename = os.path.join(subdir, filename)
        out_0 = os.path.join(out_path_0, filename)
        out_1 = os.path.join(out_path_1, filename)
        out_2 = os.path.join(out_path_2, filename)
        print("Extracting deps from {} ...".format(in_filename))
        f = open(in_filename)
        f_0 = open(out_0, "w")
        f_1 = open(out_1, "w")
        f_2 = open(out_2, "w")
        deps = f.read().split("\n")
        dep0 = []
        dep1 = []
        dep2 = []
        for i in range(len(deps)):
            if i % 4 == 0:
                dep0.append(deps[i])
            elif i % 4 == 3:
                dep2.append(deps[i])
            else:
                dep1.append(deps[i])
        # assert 0 == 1
        f_0.write("\n".join(dep0))
        f_1.write("\n".join(dep1))
        f_2.write("\n".join(dep2))
        f.close()
        f_0.close()
        f_1.close()
        f_2.close()

