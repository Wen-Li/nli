#!/usr/bin/env python3

import argparse
import csv
import os
from sklearn import metrics
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from time import time
import numpy as np
import pickle
from gensim.models import KeyedVectors
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN', 'TEL']

num_features = 300
# model = Word2Vec.load("~/Documents/glove/glove.6B.50d.txt")
print("Loading word vectors...")
model = KeyedVectors.load_word2vec_format("~/Documents/glove/glove.42B.300d.txt", binary=False)
print("Word vectors loaded!")

def load_features_and_labels(train_partition, test_partition,
                             training_feature_file,
                             test_feature_file,
                             preprocessor, vectorizer=None,
                             feature_outfile_name=None):
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

    # Read labels files
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

    training_matrix = getAvgFeatureVecs(training_files)
    test_matrix = getAvgFeatureVecs(test_files)

    with open('../pickles/word2vec_42B_train.pkl', 'wb') as f:
        pickle.dump(training_matrix, f, pickle.HIGHEST_PROTOCOL)
    with open('../pickles/word2vec_42B_dev.pkl', 'wb') as f:
        pickle.dump(test_matrix, f, pickle.HIGHEST_PROTOCOL)

    return [(training_matrix, encoded_training_labels, training_labels),
            (test_matrix, encoded_test_labels, test_labels)]

def getAvgFeatureVecs(essays):
    # Given a set of essays (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Initialize a counter
    counter = 0
    # Preallocate a 2D numpy array, for speed
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    # Loop through the essays
    for essay in essays:
        # Print a status message every 1000th essay
        if counter % 1000 == 0:
            print("essay %d of %d" % (counter, len(essays)))
        # Call the function (defined above) that makes average feature vectors
        with open(essay) as f:
            tokens = f.read().split()
            essayFeatureVecs[counter] = makeFeatureVec(tokens)
        # Increment the counter
        counter = counter + 1
    return (essayFeatureVecs)

def makeFeatureVec(words):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the essay and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        ######### LOWERCASE ################
        if word.lower() in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word.lower()])
        ####################################
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return (featureVec)

def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--train',
                   help='Name of training partition. "train" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='train')

    p.add_argument('--test',
                   help='Name of the testing partition. "dev" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='dev')

    p.add_argument('--preprocessor',
                   help='Name of directory with processed essay files. "tokenized" by default.',
                   default='tokenized')

    p.add_argument('--training_features',
                   help='Path to file containing precomputed training features. None by default. '
                        'Should be located in ../data/features/<train_partition_name>/')

    p.add_argument('--test_features',
                   help='Path to file containing precomputed test features. None by default.'
                        'Should be located in ../data/features/<test_partition_name>/')

    p.add_argument('--feature_outfile_name',
                   help='Custom name, if desired, for output feature files to be written to '
                        '../data/features/essays/<train_partition_name>/ and '
                        '../data.features/essays/<test_partition_name>. '
                        'If none provided, feature files will be named using the date and time.'
                        'If precomputed feature files are provided, this argument will be ignored.')

    p.add_argument('--predictions_outfile_name',
                   help='Custom name, if desired, for predictions file to be written to ../predictions/essays/.'
                        'If none provided, predictions file will be names using the date and time.')

    args = p.parse_args()

    training_partition_name = "train"
    test_partition_name = "dev"
    preprocessor = "tokenized"
    feature_file_train = None
    feature_file_test = None
    feature_outfile_name = None
    predictions_outfile_name = None

    #
    # Load the training and test features and labels
    #
    training_and_test_data = load_features_and_labels(training_partition_name, test_partition_name,
                                                      feature_file_train, feature_file_test,
                                                      preprocessor,
                                                      vectorizer=None,
                                                      feature_outfile_name=feature_outfile_name)
    training_matrix, encoded_training_labels, original_training_labels = training_and_test_data[0]
    test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]

    print("Train shape:", training_matrix.shape)
    print("Test shape:", test_matrix.shape)

    # Train the model
    print("Training the classifier...")
    clf = LinearSVC(C=1)

    clf.fit(training_matrix, encoded_training_labels)

    # predicted_probs = clf._predict_proba_lr(test_matrix)
    predicted = clf.predict(test_matrix)

    #
    # Display classification results
    #

    if -1 not in encoded_test_labels:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS, digits=3))
        print("\nOverall accuracy:", metrics.accuracy_score(encoded_test_labels, predicted))
    else:
        print("\nThe test set labels aren't known, cannot print accuracy report.")
