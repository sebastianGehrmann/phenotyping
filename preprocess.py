import argparse
import csv
import h5py
import re

import numpy as np
from os.path import join
from sklearn.cross_validation import train_test_split

np.random.seed(1)


class Indexer:
    def __init__(self):
        self.counter = 2
        self.d = {"<unk>": 1}
        self.rev = {}
        self._lock = False

    def convert(self, w):
        if w not in self.d:
            if self._lock:
                return self.d["<unk>"]
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >> out, k, v
        out.close()


def load_bin_vec(filename):
    """
    Loads a word2vec file and creates word2idx
    :param filename: The name of the file with word2vec vectors
    :return: word2vec dictionary, the size of embeddings and number of words in word2vec
    """
    w2v = {}
    with open(filename, 'r') as f:
        header = f.readline()
        vocab_size, emb_size = map(int, header.split())
        for line in f:
            cline = line.split()
            w2v[cline[0]] = np.array(cline[1:], dtype=np.float64)
    return w2v, emb_size, vocab_size


def parse_input_csv(filename, textfield, conditions, id_field, subj_field, chart_field):
    """
    Loads a CSV file and returns the texts as well as the condition-labels
    """
    texts = []
    target = []
    ids = []  # HAdmID
    subj = [] # subject id
    time = [] # chart time
    print "Parsing", filename
    with open(filename, 'r') as f:
        reader = csv.reader(f)  # , dialect=csv.excel_tab)
        field2id = {}
        for i, row in enumerate(reader):
            if i == 0:
                field2id = {fieldname: index for index, fieldname in enumerate(row)}
                print field2id
            else:
                texts.append("<padding> " * args.padding + row[field2id[textfield]] + " <padding>" * args.padding)
                current_targets = []
                for c in conditions:
                    current_targets.append(row[field2id[c]])
                target.append(current_targets)
                # store hospital admission ID
                # ids.append(row[field2id[id_field]])
                ids.append(i-1)
                subj.append(row[field2id[subj_field]])
                time.append(row[field2id[chart_field]])
    return texts, target, ids, subj, time


def clean_str(string):
    """
    Tokenization/string cleaning.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()  # .lower() word2vec is case sensitive


# FILE_PATHS = [# 'nursingNotesClean.csv',
#               # 'dischargeSummariesClean.csv',
#                 'AllDischargeFinal24Oct16.csv']

args = {}


def main():
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help="Source Input file", type=str)
    parser.add_argument('w2v', help="word2vec file", type=str)
    parser.add_argument('--padding', help="padding around each text", type=int, default=4)
    parser.add_argument('--batchsize', help="batchsize if you want to batch the data", type=int, default=1)
    parser.add_argument('--max_note_len', help="Cut off all notes longer than this (0 = no cutoff).", type=int,
                        default=0)
    parser.add_argument('--filename', help="File name for output file", type=str, default="data.h5")
    args = parser.parse_args()

    # LOAD THE WORD2VEC FILE
    word2vec, emb_size, v_large = load_bin_vec(args.w2v)
    print 'WORD2VEC POINTS:', v_large

    # FIELDNAMES IN CSV FILE
    textfield = 'text'
    id_field = "Hospital.Admission.ID"
    subj_field = "subject.id"
    chart_field = "chart.time"
    conditions = ['cohort', #1
                  'Obesity', #2
                  'Non.Adherence', #3
                  'Developmental.Delay.Retardation', #4
                  'Advanced.Heart.Disease', #5
                  'Advanced.Lung.Disease', #6
                  'Schizophrenia.and.other.Psychiatric.Disorders', #7
                  'Alcohol.Abuse', #8
                  'Other.Substance.Abuse', #9
                  'Chronic.Pain.Fibromyalgia', #10
                  'Chronic.Neurological.Dystrophies', #11
                  'Advanced.Cancer', #12
                  'Depression', #13
                  'Dementia', #14
                  'Unsure']

    # LOAD ALL THE DATA INTO ARRAY

    inputs, targets, ids, subj, time = parse_input_csv(args.source, textfield, conditions, id_field, subj_field, chart_field)

    print "FOUND {} DATA POINTS".format(len(inputs))

    # CONVERT ALL THE TEXT
    lbl = []
    tokenizer = Indexer()
    tokenizer.convert('padding')
    max_len_sent = 0

    for i, t in enumerate(inputs):
        current_convert = [tokenizer.convert(w) for w in clean_str(t).split()]
        max_len_sent = max(max_len_sent, len(current_convert))
        lbl.append(current_convert)
        if i % 100 == 0:
            print "CONVERTING ROW {}".format(i)
    print "MAXIMUM TEXT LENGTH IS {}".format(max_len_sent)

    # ADD PADDING TO GET TEXT INTO EQUAL LENGTH
    for sent in lbl:
        if len(sent) < max_len_sent:
            sent.extend([2] * (max_len_sent - len(sent)))
    # CUT OFF NOTE IF CUTOFF > 0.
    if args.max_note_len > 0:
        print("SHORTENING NOTES FROM {} TO {}".format(max_len_sent, args.max_note_len))
        max_len_sent = min(args.max_note_len, len(sent))
        lbl = [sent[:max_len_sent] for sent in lbl]

    # TAKING CARE OF DATA TYPE, PUT IDS WITH TEXT!
    lbl = np.column_stack([lbl, ids, subj, time])
    targets = np.array(targets, dtype=int)
    print "VOCAB SIZE {}".format(len(tokenizer.d))

    # remove the ids from the training texts
    def split_input_id(data):
        return data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

    # CONSTRUCT CORRECT EMBEDDING TABLE
    embed = np.random.uniform(-0.25, 0.25, (len(tokenizer.d), emb_size))
    unks = 0
    for key, value in tokenizer.d.iteritems():
        try:
            # -1 because of 1 indexing of word2idx (easier with torch)
            embed[value - 1] = word2vec[key]
        except:
            unks += 1
            pass
    print "{} UNKNOWN WORDS".format(unks)

    # STORE ALL THE DATA
    tokenizer.write("words.dict")

    with open("conditions.dict", 'w') as f:
        for i, c in enumerate(conditions):
            print >> f, i + 1, c

    filename = args.filename
    if args.batchsize > 1:
        # CALCULATE NUMBER OF BATCHES
        blocks = lbl.shape[0] / args.batchsize
        print "{} batches".format(blocks)
        size = blocks * args.batchsize
        print "using {} data points".format(size)
        lbl = lbl[:size]
        targets = targets[:size]

        train_size = int(round(blocks * .7) * args.batchsize)
        val_size = int(round(blocks * .1) * args.batchsize)
        test_size = size - train_size - val_size
        print "using {} train and {} validation examples".format(train_size, val_size)

        xtrain, xval, ytrain, yval = train_test_split(lbl, targets, train_size=train_size)
        xval, xtest, yval, ytest = train_test_split(xval, yval, train_size=val_size)

        # remove the ids again
        xtrain, trainids, train_subj, train_time = split_input_id(xtrain)
        xval, valids, val_subj, val_time = split_input_id(xval)
        xtest, testids, test_subj, test_time = split_input_id(xtest)

        # for some reason it needs this.
        xtrain = np.array(xtrain, dtype=int)
        xval = np.array(xval, dtype=int)
        xtest = np.array(xtest, dtype=int)
        trainids = np.array(trainids, dtype=int)
        valids = np.array(valids, dtype=int)
        testids = np.array(testids, dtype=int)
        train_subj = np.array(train_subj, dtype=int)
        val_subj = np.array(val_subj, dtype=int)
        test_subj = np.array(test_subj, dtype=int)

        train_time = np.array(train_time, dtype=int)
        val_time = np.array(val_time, dtype=int)
        test_time = np.array(test_time, dtype=int)

        print xtest.shape
        print testids.shape

        with h5py.File(filename, "w") as f:
            b = args.batchsize
            f["w2v"] = np.array(embed)

            f['train'] = np.zeros((train_size / b, b, max_len_sent), dtype=int)
            f['train_label'] = np.zeros((train_size / b, b, len(conditions)), dtype=int)
            f['test'] = np.zeros((test_size / b, b, max_len_sent), dtype=int)
            f['test_label'] = np.zeros((test_size / b, b, len(conditions)), dtype=int)
            f['val'] = np.zeros((val_size / b, b, max_len_sent), dtype=int)
            f['val_label'] = np.zeros((val_size / b, b, len(conditions)), dtype=int)
            # all the IDs
            f['train_id'] = np.zeros((train_size / b, b), dtype=int)
            f['test_id'] = np.zeros((train_size / b, b), dtype=int)
            f['val_id'] = np.zeros((train_size / b, b), dtype=int)
            f['train_subj'] = np.zeros((train_size / b, b), dtype=int)
            f['test_subj'] = np.zeros((train_size / b, b), dtype=int)
            f['val_subj'] = np.zeros((train_size / b, b), dtype=int)
            f['train_time'] = np.zeros((train_size / b, b), dtype=int)
            f['test_time'] = np.zeros((train_size / b, b), dtype=int)
            f['val_time'] = np.zeros((train_size / b, b), dtype=int)

            # STORE BATCHES
            pos = 0
            vpos = 0
            tpos = 0
            for batch in xrange(b):
                for row in range(train_size / b):
                    f['train'][row, batch] = xtrain[pos]
                    f['train_label'][row, batch] = ytrain[pos]
                    f['train_id'][row, batch] = trainids[pos]
                    f['train_subj'][row, batch] = train_subj[pos]
                    f['train_time'][row, batch] = train_time[pos]
                    pos += 1
                    if row < val_size / b:
                        # print pos, val_size, len(xval), row, batch, val_size/b, b
                        f['val'][row, batch] = xval[vpos]
                        f['val_label'][row, batch] = yval[vpos]
                        f['val_id'][row, batch] = valids[vpos]
                        f['val_subj'][row, batch] = val_subj[vpos]
                        f['val_time'][row, batch] = val_time[vpos]
                        vpos += 1
                    if row < test_size / b:
                        f['test'][row, batch] = xtest[tpos]
                        f['test_label'][row, batch] = ytest[tpos]
                        f['test_id'][row, batch] = testids[tpos]
                        f['test_subj'][row, batch] = test_subj[tpos]
                        f['test_time'][row, batch] = test_time[tpos]
                        tpos += 1

        with h5py.File(filename[:-3] + "-nobatch.h5", "w") as f:
            f["w2v"] = np.array(embed)
            f['train'] = xtrain
            f['train_label'] = ytrain
            f['train_id'] = trainids
            f['train_subj'] = train_subj
            f['train_time'] = train_time
            f['test'] = xtest
            f['test_label'] = ytest
            f['test_id'] = testids
            f['test_subj'] = test_subj
            f['test_time'] = test_time
            f['val'] = xval
            f['val_label'] = yval
            f['val_id'] = valids
            f['val_subj'] = val_subj
            f['val_time'] = val_time

    else:

        # SHUFFLE AND SPLIT DATA INTO 70-10-20

        xtrain, xval, ytrain, yval = train_test_split(lbl, targets, train_size=.7)
        xval, xtest, yval, ytest = train_test_split(xval, yval, train_size=.33)

        # remove the ids from the training texts
        xtrain, trainids, train_subj, train_time = split_input_id(xtrain)
        xval, valids, val_subj, val_time = split_input_id(xval)
        xtest, testids, test_subj, test_time = split_input_id(xtest)

        # for some reason it needs this.
        xtrain = np.array(xtrain, dtype=int)
        xval = np.array(xval, dtype=int)
        xtest = np.array(xtest, dtype=int)
        trainids = np.array(trainids, dtype=int)
        valids = np.array(valids, dtype=int)
        testids = np.array(testids, dtype=int)
        train_subj = np.array(train_subj, dtype=int)
        val_subj = np.array(val_subj, dtype=int)
        test_subj = np.array(test_subj, dtype=int)
        train_time = np.array(train_time, dtype=int)
        val_time = np.array(val_time, dtype=int)
        test_time = np.array(test_time, dtype=int)

        with h5py.File(filename, "w") as f:
            f["w2v"] = np.array(embed)
            f['train'] = xtrain
            f['train_label'] = ytrain
            f['train_id'] = trainids
            f['train_subj'] = train_subj
            f['train_time'] = train_time
            f['test'] = xtest
            f['test_label'] = ytest
            f['test_id'] = testids
            f['test_subj'] = test_subj
            f['test_time'] = test_time
            f['val'] = xval
            f['val_label'] = yval
            f['val_id'] = valids
            f['val_subj'] = val_subj
            f['val_time'] = val_time


if __name__ == '__main__':
    main()
