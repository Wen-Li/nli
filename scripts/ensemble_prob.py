import pickle
import numpy as np
import os
from sklearn import metrics
import csv

path_prefix = "../best_probabilities/test/"
test_partition_name = "test"
predictions_file_name = "log_sum_probs_test.csv"
# predictions_file_name = "char3to9_dev.csv"
encoded_test_labels = np.array(pickle.load(open("../pickles/encoded_test_labels.pkl", "rb")))

sum_probs = np.zeros([1100, 11])
prob_files = [
    "char_tokenized_3_10000.pkl",
    "char_tokenized_4_30000.pkl",
    "char_tokenized_5_30000.pkl",
    "char_tokenized_6_30000.pkl",
    "char_tokenized_7_50000.pkl",
    # "char_tokenized_8_50000.pkl",
    # "char_tokenized_9_50000.pkl",
    "word_tokenized_1_20000.pkl",
    "word_tokenized_2_50000.pkl",
    "word_tokenized_3_50000.pkl",
    "word_stemmed_1_10000.pkl",
    "word_stemmed_2_30000.pkl",
    "word_stemmed_3_30000.pkl",

    # "word_parsed_1_50000.pkl",

    # "word_tagged_1_all.pkl",
    # "word_tagged_2_all.pkl",
    # "word_tagged_3_all.pkl",
    # "word_function_words_1_all.pkl",
    "word_function_words_2_all.pkl",
    "word_function_words_3_all.pkl",

    "word_dep0_1_30000.pkl",
    "word_dep1_1_30000.pkl",
    "word_dep2_1_4000.pkl"

    # "wordvec_300.pkl"

]

sum_probs = np.zeros([1100, 11])
for fn in prob_files:
    p = np.array(pickle.load(open(path_prefix + fn, "rb")))
    sum_probs = np.add(sum_probs, np.log(p))

print(sum_probs.shape)
print(sum_probs[0])

predicted = np.argmax(sum_probs, axis=1)

# Write Predictions File

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN','TEL']



# evaluation
def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))
    return

if test_partition_name == "dev":
    print("\nConfusion Matrix:\n")
    cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
    pretty_print_cm(cm, CLASS_LABELS)
    print("\nClassification Results:\n")
    print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS, digits=3))
    print("\nOverall accuracy:", metrics.accuracy_score(encoded_test_labels, predicted))
else:
    print("\nThe test set labels aren't known, cannot print accuracy report.")
    
labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
                    .format(script_dir=SCRIPT_DIR, test=test_partition_name))

outfile = '{script_dir}/../predictions/essays/{pred_file}'.format(script_dir=SCRIPT_DIR,
                                                                  pred_file=predictions_file_name)
with open(outfile, 'w+', newline='', encoding='utf8') as output_file:
    file_writer = csv.writer(output_file)
    with open(labels_file_path, encoding='utf-8') as labels_file:
        label_rows = [row for row in csv.reader(labels_file)]
        label_rows[0].append('prediction')
        for i, row in enumerate(label_rows[1:]):
            encoded_prediction = int(predicted[i])
            prediction = CLASS_LABELS[encoded_prediction]
            row.append(prediction)
    file_writer.writerows(label_rows)

print("Predictions written to", outfile.replace(SCRIPT_DIR, '')[1:], "(%d lines)" % len(predicted))

