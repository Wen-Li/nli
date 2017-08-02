#!/usr/bin/env python3

"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate the features from the tokenized essays:
        $ python n_grams.py [--train ] [--test] [--preprocessor]

    Run with precomputed features:
        $ python n_grams.py [--train] [--test dev] [--preprocessor] --training_features path/to/train/featurefile --test_features /path/to/test/featurefile
"""
import numpy as np
import csv
import os
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import time
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score


def load_features_and_labels(train_partition, test_partition):
    train_labels_path = "{script_dir}/../data/labels/{train}/labels.{train}.csv".format(train=train_partition,
                                                                                        script_dir=SCRIPT_DIR)
    train_data_path = "{script_dir}/../data/essays/{}/tokenized/".format(train_partition, script_dir=SCRIPT_DIR)
    test_labels_path = "{script_dir}/../data/labels/{test}/labels.{test}.csv".format(test=test_partition,
                                                                                     script_dir=SCRIPT_DIR)
    test_data_path = "{script_dir}/../data/essays/{}/tokenized".format(test_partition, script_dir=SCRIPT_DIR)

    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (train_data_path, "training data directory"),
                                (test_labels_path, "testing labels file"),
                                (test_data_path, "testing data directory")]
    for path_, path_descriptor in path_and_descriptor_list:
        if not os.path.exists(path_):
            raise Exception("Could not find {desc}: {pth}".format(desc=path_descriptor, pth=path_))

    # Read labels files. If feature files provided, `training_files` and `test_files` below will be ignored
    with open(train_labels_path) as train_labels_f, open(test_labels_path) as test_labels_f:
        essay_path_train = '{script_dir}/../data/essays/{train}/{preproc}'.format(script_dir=SCRIPT_DIR,
                                                                                  train=train_partition,
                                                                                  preproc=preprocessor)
        essay_path_test = '{script_dir}/../data/essays/{test}/{preproc}'.format(script_dir=SCRIPT_DIR,
                                                                                test=test_partition,
                                                                                preproc=preprocessor)
        training_files, training_labels = zip(
            *[(os.path.join(essay_path_train, row['test_taker_id'] + '.txt'), row['L1'])
              for row in csv.DictReader(train_labels_f)])
        test_files, test_labels = zip(*[(os.path.join(essay_path_test, row['test_taker_id'] + '.txt'), row['L1'])
                                        for row in csv.DictReader(test_labels_f)])
    print("Found {} text files in {} and {} in {}"
          .format(len(training_files), train_partition, len(test_files), test_partition))
    print("Loading training and testing data from {} & {}".format(train_partition, test_partition))

    # encode train and test labels: char to int
    encoded_training_labels = [CLASS_LABELS.index(label) for label in training_labels]
    encoded_test_labels = [CLASS_LABELS.index(label) for label in test_labels]

    # save encoded train and test labels
    # with open('../pickles/encoded_training_labels.pkl', 'wb') as f:
    #     pickle.dump(encoded_training_labels, f, pickle.HIGHEST_PROTOCOL)
    # with open('../pickles/encoded_test_labels.pkl', 'wb') as f:
    #     pickle.dump(encoded_test_labels, f, pickle.HIGHEST_PROTOCOL)

    # combine train and dev files
    file_list = list(training_files) + list(test_files)
    vectorizer = TfidfVectorizer(input="filename",
                                 analyzer=unit,
                                 ngram_range=unit_range,
                                 token_pattern=unit_pattern,
                                 min_df=2)
    tfidf = vectorizer.fit_transform(file_list)
    n = len(vectorizer.get_feature_names())
    print("\n".join(vectorizer.get_feature_names()[int(n/2):int(n/2 + 10)]))
    training_matrix = tfidf[:len(training_files)]
    test_matrix = tfidf[len(training_files):]

    return [(training_matrix, encoded_training_labels, training_labels),
            (test_matrix, encoded_test_labels, test_labels)]


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))

def sk(X, Y):
    return(np.dot(X, Y))

if __name__ == '__main__':

    t0 = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN', 'TEL']

    ################## N-GRAM SETTINGS ###############
    unit = u"char"

    unit_pattern = u"\S+"
    # unit_pattern = u"_(\S+)"    # only for POS tags

    # preprocessor = "tagged"
    # preprocessor = "function_words"
    # preprocessor = "parsed"
    preprocessor = "tokenized"

    training_partition_name = "train"
    test_partition_name = "dev"

    # Load the training and test features and labels

    for unit_range in [(3,3)]:
        for num_features in [10000]:
            training_and_test_data = load_features_and_labels(training_partition_name, test_partition_name)
            training_matrix, encoded_training_labels, original_training_labels = training_and_test_data[0]
            test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]

            print("Train and test shape:", training_matrix.shape, test_matrix.shape)

            # feature normalization
            # transformer = Normalizer()
            # training_matrix = transformer.fit_transform(training_matrix)
            # test_matrix = transformer.fit_transform(test_matrix)

            # feature selection
            if num_features != "all":
                feature_selection = SelectKBest(chi2, k=num_features)
                training_matrix = feature_selection.fit_transform(training_matrix, encoded_training_labels)
                test_matrix = feature_selection.transform(test_matrix)

            print("After feature selection:", training_matrix.shape, test_matrix.shape)

            # Train the model
            print("Training the classifier...")
            clf = SVC(kernel=sk, probability=True)
            # clf = MLPClassifier(hidden_layer_sizes=(100,))

            # 10-fold cv on train and dev
            # cv_scores = cross_val_score(clf, training_matrix, encoded_training_labels, cv=10)
            # print("Cross validation (scores on 10-fold, avg, std):")
            # print(cv_scores, np.mean(cv_scores), np.std(cv_scores))
            train_probs = cross_val_predict(clf, training_matrix, encoded_training_labels,
                                            method="predict_proba", cv=10)
            train_pred = np.argmax(train_probs, axis=1)
            print("{} {} {} Accuracy on train:".format(unit, unit_range, preprocessor),
                  metrics.accuracy_score(encoded_training_labels, train_pred))

            # evaluation
            # if -1 not in encoded_training_labels:
            #     print("\nConfusion Matrix:\n")
            #     cm = metrics.confusion_matrix(encoded_training_labels, predicted).tolist()
            #     pretty_print_cm(cm, CLASS_LABELS)
            #     print("\nClassification Results:\n")
            #     print(metrics.classification_report(encoded_training_labels, predicted, target_names=CLASS_LABELS, digits=3))
            #     print("\nOverall accuracy:", metrics.accuracy_score(encoded_training_labels, predicted))
            # else:
            #     print("\nThe test set labels aren't known, cannot print accuracy report.")

            # predict probabilities on test
            clf.fit(training_matrix, encoded_training_labels)
            test_probs = clf.predict_proba(test_matrix)
            if test_partition_name == "dev":
                test_pred = np.argmax(test_probs, axis=1)
                print("Accuracy on dev:", metrics.accuracy_score(encoded_test_labels, test_pred))

            # save probabilities
            with open('../probabilities/{}/sk_{}_{}_{}_{}.pkl'
                              .format(training_partition_name,
                                      unit, preprocessor,
                                      str(unit_range[0]), str(num_features)), 'wb') as f:
                pickle.dump(train_probs, f, pickle.HIGHEST_PROTOCOL)
            with open('../probabilities/{}/sk_{}_{}_{}_{}.pkl'
                              .format(test_partition_name,
                                      unit, preprocessor,
                                      str(unit_range[0]), str(num_features)), 'wb') as f:
                pickle.dump(test_probs, f, pickle.HIGHEST_PROTOCOL)

            #
            # Display classification results
            #

            # if -1 not in encoded_test_labels:
            #     print("\nConfusion Matrix:\n")
            #     cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
            #     pretty_print_cm(cm, CLASS_LABELS)
            #     print("\nClassification Results:\n")
            #     print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS, digits=3))
            #     print("\nOverall accuracy:", metrics.accuracy_score(encoded_test_labels, predicted))
            # else:
            #     print("\nThe test set labels aren't known, cannot print accuracy report.")

            print("Executed in %.4s seconds." % (time.time() - t0))
