import pickle
import numpy as np
from sklearn import metrics

# char8 = np.array(pickle.load(open("../probabilities/train_dev/char_8_350000.pkl", "rb")))
# char9 = np.array(pickle.load(open("../probabilities/train_dev/char_9_350000.pkl", "rb")))

# result = np.add(char9, char9)
result = np.array(pickle.load(open("../best_probabilities/dev/word_lemmatized_1_20000.pkl", "rb")))
predicted = np.argmax(result, axis=1)

CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN','TEL']

encoded_train_labels = np.array(pickle.load(open("../pickles/encoded_test_labels.pkl", "rb")))

def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


if -1 not in encoded_train_labels:
    print("\nConfusion Matrix:\n")
    cm = metrics.confusion_matrix(encoded_train_labels, predicted).tolist()
    pretty_print_cm(cm, CLASS_LABELS)
    print("\nClassification Results:\n")
    print(metrics.classification_report(encoded_train_labels, predicted, target_names=CLASS_LABELS, digits=3))
    print("\nOverall accuracy:", metrics.accuracy_score(encoded_train_labels, predicted))
else:
    print("\nThe test set labels aren't known, cannot print accuracy report.")