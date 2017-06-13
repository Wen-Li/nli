import pickle

word1 = pickle.load(open("../pickles/word_2_probs_50000.pkl", "rb"))

CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN','TEL']


if -1 not in encoded_test_labels:
    print("\nConfusion Matrix:\n")
    cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
    pretty_print_cm(cm, CLASS_LABELS)
    print("\nClassification Results:\n")
    print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS, digits=3))
    print("\nOverall accuracy:", metrics.accuracy_score(encoded_test_labels, predicted))

for i in range(3):
    print(word1[i])