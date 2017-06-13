#!/usr/bin/env python3

"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate the features from the tokenized essays:
        $ python n_grams.py [--train ] [--test] [--preprocessor]

    Run with precomputed features:
        $ python n_grams.py [--train] [--test dev] [--preprocessor] --training_features path/to/train/featurefile --test_features /path/to/test/featurefile
"""
import argparse
import csv
import os
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

from sklearn.feature_selection import SelectKBest, chi2
from time import time
from sklearn.decomposition import NMF
import numpy as np
import pickle

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_LABELS = ['FRE', 'GER', 'ITA', 'SPA', 'ARA', 'TUR', 'CHI', 'JPN', 'KOR', 'HIN','TEL']

unit = u"word"
unit_range = (2, 2)
num_features = 50000

def load_features_and_labels(train_partition, test_partition, training_feature_file="../data/features/essays/train/train-2017-05-27-15.39.30.features",
                             test_feature_file="../data/features/essays/dev/dev-2017-05-27-15.39.34.features", preprocessor='tokenized', vectorizer=None,
                             feature_outfile_name=None):

    train_labels_path = "{script_dir}/../data/labels/{train}/labels.{train}.csv".format(train=train_partition, script_dir=SCRIPT_DIR)
    train_data_path = "{script_dir}/../data/essays/{}/tokenized/".format(train_partition, script_dir=SCRIPT_DIR)
    test_labels_path = "{script_dir}/../data/labels/{test}/labels.{test}.csv".format(test=test_partition, script_dir=SCRIPT_DIR)
    test_data_path = "{script_dir}/../data/essays/{}/tokenized".format(test_partition, script_dir=SCRIPT_DIR)

    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (train_data_path, "training data directory"),
                                (test_labels_path, "testing labels file"),
                                (test_data_path, "testing data directory")]
    for path_, path_descriptor in path_and_descriptor_list:
        if not os.path.exists(path_):
            raise Exception("Could not find {desc}: {pth}".format(desc=path_descriptor, pth=path_))

    #  Read labels files. If feature files provided, `training_files` and `test_files` below will be ignored
    with open(train_labels_path) as train_labels_f, open(test_labels_path) as test_labels_f:
        essay_path_train = '{script_dir}/../data/essays/{train}/{preproc}'.format(script_dir=SCRIPT_DIR, train=train_partition, preproc=preprocessor)
        essay_path_test = '{script_dir}/../data/essays/{test}/{preproc}'.format(script_dir=SCRIPT_DIR, test=test_partition, preproc=preprocessor)
        training_files, training_labels = zip(*[(os.path.join(essay_path_train, row['test_taker_id'] + '.txt'), row['L1'])
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
    with open('../pickles/encoded_training_labels.pkl', 'wb') as f:
        pickle.dump(encoded_training_labels, f, pickle.HIGHEST_PROTOCOL)
    with open('../pickles/encoded_test_labels.pkl', 'wb') as f:
        pickle.dump(encoded_test_labels, f, pickle.HIGHEST_PROTOCOL)

    # combine train and test files
    file_list = list(training_files) + list(test_files)
    vectorizer = TfidfVectorizer(input="filename",
                                 analyzer=unit, ngram_range=unit_range,
                                 token_pattern=u"\S+",
                                 min_df=2)
    tfidf = vectorizer.fit_transform(file_list)
    training_matrix = tfidf[:len(training_files)]
    test_matrix = tfidf[len(training_files):]

    return [(training_matrix, encoded_training_labels, training_labels),
            (test_matrix, encoded_test_labels, test_labels)]


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))

def print_top_words(model, feature_names, n_top_words):
    n = 0
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        n += 1
        if n > 50:
            break
    print()
    return

def feature_processing(training_matrix, test_matrix):
    # Normalize frequencies to unit length
    transformer = Normalizer()
    training_matrix = transformer.fit_transform(training_matrix)
    test_matrix = transformer.fit_transform(test_matrix)

    # feature selection
    feature_selection = SelectKBest(chi2, k=num_features)
    training_matrix = feature_selection.fit_transform(training_matrix, encoded_training_labels)
    test_matrix = feature_selection.transform(test_matrix)

    return(training_matrix, test_matrix)

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

    training_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    feature_file_train = args.training_features
    feature_file_test = args.test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    #
    # Load the training and test features and labels
    #
    training_and_test_data = load_features_and_labels(training_partition_name, test_partition_name, feature_file_train, 
                                                      feature_file_test, feature_outfile_name=feature_outfile_name)
    training_matrix, encoded_training_labels, original_training_labels = training_and_test_data[0]
    test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]

    print("Train shape:", training_matrix.shape)
    print("Test shape:", test_matrix.shape)

    # feature pre-processing
    # training_matrix, test_matrix = feature_processing(training_matrix, test_matrix)

    # feature normalization
    # transformer = Normalizer()
    # training_matrix = transformer.fit_transform(training_matrix)
    # test_matrix = transformer.fit_transform(test_matrix)

    # Train the model
    print("Training the classifier...")
    clf = LinearSVC(C=0.8)

    clf.fit(training_matrix, encoded_training_labels)

    # predicted_probs = clf._predict_proba_lr(test_matrix)
    predicted = clf.predict(test_matrix)

    # save predicted probabilities
    # with open('../pickles/nmf_50_1000.pkl', 'wb') as f:
        # pickle.dump(predicted_probs, f, pickle.HIGHEST_PROTOCOL)

    #
    # Write Predictions File
    #

    # labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
    #                     .format(script_dir=SCRIPT_DIR, test=test_partition_name))
    #
    # predictions_file_name = (strftime("predictions-%Y-%m-%d-%H.%M.%S.csv")
    #                          if predictions_outfile_name is None
    #                          else predictions_outfile_name)
    #
    # outfile = '{script_dir}/../predictions/essays/{pred_file}'.format(script_dir=SCRIPT_DIR, pred_file=predictions_file_name)
    # with open(outfile, 'w+', newline='', encoding='utf8') as output_file:
    #     file_writer = csv.writer(output_file)
    #     with open(labels_file_path, encoding='utf-8') as labels_file:
    #         label_rows = [row for row in csv.reader(labels_file)]
    #         label_rows[0].append('prediction')
    #         for i, row in enumerate(label_rows[1:]):
    #             encoded_prediction = int(predicted[i])
    #             prediction = CLASS_LABELS[encoded_prediction]
    #             row.append(prediction)
    #     file_writer.writerows(label_rows)
    #
    # print("Predictions written to", outfile.replace(SCRIPT_DIR, '')[1:], "(%d lines)" % len(predicted))

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

