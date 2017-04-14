import math
import pickle
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix


stopwords = '、。〈〉《》︿！＃＄％＆（）＊＋，０１２３４５６７８９：；＜＞？＠［］｛｜｝～￥'


class tfidfvectorizer():
    def __init__(self, stopwords=stopwords):
        self.vocabulary = defaultdict()
        self.vocabulary_size = 0
        self.stopwords = stopwords
        self.df = [0] * 1000000
        self.raw_tf = []
        self.N = 0

    def _to_sparse_matrix(self, X):
        X = np.array(X).T
        row = X[0]
        col = X[1]
        freq = X[2]

        return csr_matrix((freq, (row, col)))

    def tfidf(self, doc_i, tok_j, freq):
        k1 = 1.6
        b = 0.9
        freq = freq * (k1 + 1) / (freq + k1 * (1 - b + b * self.doclen[doc_i] / self.avgdoclen)) * self.idf[tok_j]
        return (doc_i, tok_j, freq)

    def transform(self, X):
        """Convert raw tf to desired weight given list of (DOC_ID, TOK_INDEX, FREQ) tuples."""
        Y = [0] * len(X)

        self.avgdoclen = sum(self.doclen) / self.N
        self.df = np.array(self.df)
        self.idf = np.log((self.N - self.df + 0.5) / (self.df + 0.5))
        for i, (doc_i, tok_j, freq) in enumerate(X):
            Y[i] = self.tfidf(doc_i, tok_j, freq)

        return self._to_sparse_matrix(Y)

    def vectorize(self, documents):
        self.N = len(documents)
        self.doclen = [0] * self.N
        for doc_i, doc in enumerate(documents):
            self.doclen[doc_i] = len(doc)
            tokens = [tok for tok in doc.split() if tok not in stopwords]

            doc_term = defaultdict(int)

            for tok in tokens:
                tok_ind = self.vocabulary.setdefault(tok, self.vocabulary_size)
                if tok_ind == self.vocabulary_size:
                    self.vocabulary_size += 1

                doc_term[tok_ind] = doc_term[tok_ind] + 1

            for tok_ind in doc_term.keys():
                self.df[tok_ind] += 1

            # List of (DOC_ID, TOK_INDEX, FREQ) tuples.
            self.raw_tf.extend([(doc_i, tok_j, freq) for tok_j, freq in doc_term.items()])

        return self.transform(self.raw_tf)

    def save(self, filename='raw_tf.p'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.vocabulary_size, fp)
            pickle.dump(self.N, fp)
            pickle.dump(self.df, fp)
            pickle.dump(self.raw_tf, fp)
            pickle.dump(self.doclen, fp)

    def load(self, filename='raw_tf.p'):
        with open(filename, 'rb') as fp:
            self.vocabulary_size = pickle.load(fp)
            self.N = pickle.load(fp)
            self.df = pickle.load(fp)
            self.raw_tf = pickle.load(fp)
            self.doclen = pickle.load(fp)
