from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

X_train = np.array(pickle.load(open("../pickles/word2vec_42B_train.pkl", "rb")))
y_train = np.array(pickle.load(open("../pickles/encoded_training_labels.pkl", "rb")))
X_test = np.array(pickle.load(open("../pickles/word2vec_42B_dev.pkl", "rb")))
y_test = np.array(pickle.load(open("../pickles/encoded_test_labels.pkl", "rb")))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

clf = MLPClassifier(hidden_layer_sizes=(300, 50))

print("Started training ...")
clf.fit(X_train, y_train)
print("Finished training.")

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred).tolist()
print(cm)

print(classification_report(y_test, y_pred, digits=3))
print("\nOverall accuracy:", accuracy_score(y_test, y_pred))
