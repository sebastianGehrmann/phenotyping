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

    def load(self, dictfile):
        with open(dictfile, 'r') as f:
            for line in f:
                self.convert(line.split()[0])
        self.lock()
        print("SUCCESSFULLY LOADED DICTIONARY FILE")

def parse_input_csv(filename, textfield):
    """
    Loads a CSV file and returns the texts as well as the condition-labels
    """
    texts = []
    print "Parsing", filename
    with open(filename, 'r') as f:
        reader = csv.reader(f)#, dialect=csv.excel_tab)
        field2id = {}
        for i, row in enumerate(reader):
            if i == 0:
                field2id = {fieldname: index for index, fieldname in enumerate(row)}
                print field2id
            else:
                texts.append("<padding> " * args.padding + row[field2id[textfield]] + " <padding>" * args.padding)

    return texts


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
    parser.add_argument('dictfile', help="Dictionary file", type=str)
    parser.add_argument('--padding', help="padding around each text", type=int, default=4)
    parser.add_argument('--batchsize', help="batchsize if you want to batch the data", type=int, default=1)
    parser.add_argument('--max_note_len', help="Cut off all notes longer than this (0 = no cutoff).", type=int, default=0)
    parser.add_argument('--filename', help="File name for output file", type=str, default="testdata.h5")
    args = parser.parse_args()


    # FIELDNAMES IN CSV FILE
    textfield = 'text'

    # LOAD ALL THE DATA INTO ARRAY

    inputs = parse_input_csv(args.source, textfield)

    print "FOUND {} DATA POINTS".format(len(inputs))

    # CONVERT ALL THE TEXT
    lbl = []
    tokenizer = Indexer()
    tokenizer.load(args.dictfile)
    max_len_sent = args.max_note_len

    for i, t in enumerate(inputs):
        current_convert = [tokenizer.convert(w) for w in clean_str(t).split()]
        current_convert = current_convert[:max_len_sent]
        lbl.append(current_convert)
        if i % 100 == 0:
            print "CONVERTING ROW {}".format(i)
    print "MAXIMUM TEXT LENGTH IS {}".format(max_len_sent)


    # ADD PADDING TO GET TEXT INTO EQUAL LENGTH
    for sent in lbl:
        if len(sent) < max_len_sent:
            sent.extend([2] * (max_len_sent - len(sent)))

    # TAKING CARE OF DATA TYPE
    lbl = np.array(lbl, dtype=int)


    filename = args.filename
    if args.batchsize > 1:
        #CALCULATE NUMBER OF BATCHES
        blocks = lbl.shape[0] / args.batchsize
        print "{} batches".format(blocks)
        size = blocks * args.batchsize
        print "using {} data points".format(size)
        lbl = lbl[:size]

        train_size = int(blocks * args.batchsize)
        print "using {} examples".format(train_size)

        with h5py.File(filename, "w") as f:
            b = args.batchsize

            f['test'] = np.zeros((train_size/b, b, max_len_sent), dtype=int)
            #STORE BATCHES
            pos = 0
            vpos = 0
            tpos = 0
            #DIFFERENT THAN USUAL PREPROCESSING!!!
            for row in range(train_size / b):
                for batch in xrange(b):
                    f['test'][row, batch] = lbl[pos]
                    pos+=1

    else:
        with h5py.File(filename, "w") as f:
            f['test'] = lbl


if __name__ == '__main__':
    main()
