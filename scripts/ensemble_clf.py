import pickle
import numpy as np
import os
from sklearn import metrics
import csv
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN', 'TEL']

train_partition_name = "train_dev"
test_partition_name = "test"

# train_prefix = "../best_probabilities/{}/".format(train_partition_name)
test_prefix = "../best_probabilities/{}/".format(test_partition_name)

predictions_file_name = "baseline_word_1.csv"

encoded_train_labels = np.array(pickle.load(open("../pickles/encoded_{}_labels.pkl"
                                                 .format(train_partition_name), "rb")))
encoded_test_labels = np.array(pickle.load(open("../pickles/encoded_test_labels.pkl", "rb")))

prob_files = (
    "word_tokenized_1_all.pkl",
    # "char_tokenized_3_10000.pkl",
    # "char_tokenized_4_30000.pkl",
    # "char_tokenized_5_30000.pkl",
    # "char_tokenized_6_30000.pkl",
    # "char_tokenized_7_50000.pkl",
    # "char_tokenized_8_50000.pkl",
    # "char_tokenized_9_50000.pkl",
    # "word_tokenized_1_20000.pkl",
    # "word_tokenized_2_50000.pkl",
    # "word_tokenized_3_50000.pkl",
    # "word_stemmed_1_10000.pkl",
    # "word_stemmed_2_30000.pkl",
    # "word_stemmed_3_30000.pkl",
    # "word_lemmatized_1_20000.pkl",
    # "word_lemmatized_2_50000.pkl",
    # "word_lemmatized_3_50000.pkl",

    # "word_parsed_1_80000.pkl",

    # "word_tagged_1_all.pkl",
    # "word_tagged_2_all.pkl",
    # "word_tagged_3_all.pkl",
    # "word_function_words_1_all.pkl",
    # "word_function_words_2_all.pkl",
    # "word_function_words_3_all.pkl",
    #
    # "word_dep0_1_30000.pkl",
    # "word_dep1_1_30000.pkl",
    # "word_dep2_1_4000.pkl",

    # "wordvec_300.pkl"
)

def concat_prob_matrix(path_prefix):
    probs = []
    for fn in prob_files:
        p = np.array(pickle.load(open(path_prefix + fn, "rb")))
        probs.append(p)
    combined_probs = np.hstack(np.array(probs))
    return(combined_probs)

# train_matrix = concat_prob_matrix(train_prefix)
test_matrix = concat_prob_matrix(test_prefix)

# print(train_matrix.shape, test_matrix.shape)
# print(train_matrix[0])
# print(test_matrix[0])

# evaluate on train

# run clf for 20 times, add up probs
sum_probs = np.zeros([1100, 11])
for i in range(10):
    print("i =", i)
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, tol=0.00001)
    # clf = LinearSVC(C=0.8)
    # clf = LinearDiscriminantAnalysis()
    # if i % 2 == 0:
    # if True:
    #     train_pred = cross_val_predict(clf, train_matrix, encoded_train_labels, cv=5)
    #     print("Accuracy on train:", metrics.accuracy_score(encoded_train_labels, train_pred))
    # clf.fit(train_matrix, encoded_train_labels)
    # predicted = clf.predict(test_matrix)
    ###### MLP
    # predicted_prob = clf.predict_proba(test_matrix)
    ###### SVM
    # predicted_prob = clf._predict_proba_lr(test_matrix)

    # sum_probs = np.add(sum_probs, predicted_prob)
    # print(predicted_prob.shape)
    # print(encoded_test_labels.shape)

predicted = np.argmax(sum_probs, axis=1)
predicted = np.argmax(test_matrix, axis=1)


# BAGGING
# clf = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, tol=0.0001),
#                             n_estimators=50,
#                             max_samples=100, bootstrap=True, n_jobs=-1
# )
# clf.fit(train_matrix, encoded_train_labels)
# predicted = clf.predict(test_matrix)

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
