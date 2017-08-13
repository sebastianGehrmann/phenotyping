'''
Text classification
'''

from __future__ import print_function
from __future__ import division

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import sklearn.multiclass
import sklearn.svm
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.externals.joblib
import sklearn.metrics
import sklearn.feature_extraction.text
from nltk.tokenize import TreebankWordTokenizer
import utils

import argparse
import h5py

if utils.is_python3():
    import configparser as ConfigParser
else:
    import ConfigParser

import scipy

print('The NumPy version is {0}.'.format(np.version.version))
print('The scikit-learn version is {0}.'.format(sklearn.__version__))
print('The SciPy version is {0}\n'.format(scipy.version.full_version))  # requires SciPy >= 0.16.0


def extract_features(X, val_X, test_X, max_ngram_size=2):
    print('max_ngram_size: {0}'.format(max_ngram_size))

    def make_text_list(X):

        return [' '.join([str(num) for num in x]) for x in X]

    print("Creating N-grams")
    X_texts = make_text_list(X)
    print("Creating N-grams for valid")
    val_X_texts = make_text_list(val_X)
    print("Creating N-grams for test")
    test_X_texts = make_text_list(test_X)
    all_texts = X_texts + val_X_texts + test_X_texts

    vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, max_ngram_size),
                                                           tokenizer=TreebankWordTokenizer().tokenize)
    vect.fit(all_texts)  # build ngram dictionary
    X_features = vect.transform(X_texts)
    val_X_features = vect.transform(val_X_texts)
    test_X_features = vect.transform(test_X_texts)

    #print('features.shape: {0}'.format(features.shape))
    #np.savez(features_filepath, features=features)
    np.savez("converted/X_features-"+str(max_ngram_size)+".npz", features=X_features)
    np.savez("converted/val_X_features-"+str(max_ngram_size)+".npz", features=val_X_features)
    np.savez("converted/test_X_features-"+str(max_ngram_size)+".npz", features=test_X_features)

    return X_features, val_X_features, test_X_features

def make_predictions(X, Y, val_X, val_Y, test_X, test_Y, s):
    # classifier = sklearn.svm.LinearSVC(n_jobs=2)
    classifier = sklearn.linear_model.LogisticRegression(n_jobs=2)
    # classifier = sklearn.naive_bayes.GaussianNB()
    # classifier = sklearn.svm.SVC(probability=True) # hard to scale to dataset with more than a couple of 10000 samples.

    total_data = X.shape[0]
    #subset for analysis
    # X = X[:round(total_data*s), :]
    # Y = Y[:round(total_data*s)]

    # print(s)
    # print("WITH {} TRAINING POINTS".format(X.shape[0]))

    classifier.fit(X, Y)
    test_y_hat = classifier.predict(test_X)
    results = sklearn.metrics.classification_report(test_Y, test_y_hat, digits=3)
    print('results: {0}'.format(results))
    accuracy_score = sklearn.metrics.accuracy_score(test_Y, test_y_hat)
    print('accuracy_score: {0}'.format(accuracy_score))

    roc_auc_score = sklearn.metrics.roc_auc_score(test_Y, test_y_hat)
    print('roc_auc_score: {0}'.format(roc_auc_score))

    # Plot ROC
    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(test_Y,
                                                                                    test_y_hat)
    return accuracy_score, roc_auc_score

conditions = ['cohort', #0
              'Obesity', #1
              'Non.Adherence', #2
              'Developmental.Delay.Retardation', #3
              'Advanced.Heart.Disease', #4
              'Advanced.Lung.Disease', #5
              'Schizophrenia.and.other.Psychiatric.Disorders', #6
              'Alcohol.Abuse', #7
              'Other.Substance.Abuse', #8
              'Chronic.Pain.Fibromyalgia', #9
              'Chronic.Neurological.Dystrophies', #10
              'Advanced.Cancer', #11
              'Depression', #12
              'Dementia', #13
              'Unsure'] #14


def main():
    '''
    This is the main function
    '''
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help="The input hdf5 file", type=str, default='data_no_batch.h5')
    parser.add_argument('--ngram', help="Maximum ngram length", type=int, default=3)
    args = parser.parse_args()

    all_accs = []
    all_aucs = []

    with h5py.File(args.data, "r") as f:
        if not os.path.isfile(os.path.join("converted/X_features-"+str(args.ngram)+".npz")):
            print('Computing features.')
            train_x = f["train"][:]
            valid_x = f["val"][:]
            test_x = f["test"][:]

            X_features, val_X_features, test_X_features = extract_features(train_x, valid_x, test_x, args.ngram)
        else:
            print('Skipping Build of Ngrams: model already exists.')
            X_features = dict(np.load(os.path.join("converted", "X_features-"+str(args.ngram)+".npz")))['features'].item()
            val_X_features = dict(np.load(os.path.join("converted", "val_X_features-"+str(args.ngram)+".npz")))['features'].item()
            test_X_features = dict(np.load(os.path.join("converted", "test_X_features-"+str(args.ngram)+".npz")))['features'].item()




        for index, condition in enumerate(conditions):
            # for dataset_filepath in sorted(glob.glob(os.path.join(data_folder_formatted, 'icu_frequent_flyers_cohort.npz'))):

            print('Current Condition: {0}'.format(condition))

            train_y = f["train_label"][:,index]
            valid_y = f["val_label"][:,index]
            test_y = f["test_label"][:,index]
            current_accs = []
            current_aucs = []
            #for subset in xrange(1,21):
            #    s = subset/float(20)
            acc, auc = make_predictions(X_features, train_y, val_X_features, valid_y, test_X_features, test_y, 1)#s)
            current_accs.append(acc)
            current_aucs.append(auc)
            print('\n')
                # break
            #all_accs.append(current_accs)
            #all_aucs.append(current_aucs)

    # with h5py.File("sufficient_data_eval.h5", 'w') as f:
    #     f['accs'] = np.array(all_accs)
    #     f['aucs'] = np.array(all_aucs)

if __name__ == "__main__":
    main()
    # cProfile.run('main()') # if you want to do some profiling
