import jieba
import os
import os.path
import argparse
import time

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def cut(x):
    if not x:
        x = ' '
    return ' '.join(jieba.cut(x)) + ' '


def doc2dict(file):
    e = ET.parse(file).getroot()
    text = ''
    for x in e.find('doc/text').iter():
        text += x.text

    d = {
        'id': e.find('doc/id').text,
        'title': cut(e.find('doc/title').text),
        'date': e.find('doc/date').text,
        'text': cut(text)}

    return d


def query2dicts(file):
    e = ET.parse(file).getroot()
    text = ''
    d_list = []
    for x in e.findall('topic'):
        d_list.append({
            'number': x.find('number').text,
            'title': cut(x.find('title').text),
            'question': cut(x.find('question').text),
            'concepts': cut(x.find('concepts').text),
            'narrative': cut(x.find('narrative').text)})

    return d_list


def make_corpus(doc, query):
    doc_content = doc.title.apply(lambda x: x + ' ') + doc.text
    query_content = query.concepts.apply(lambda x: x + ' ') + query.title

    corpus = pd.concat((doc_content, query_content)).as_matrix()

    return corpus


def calc_AP(ret, ans):
    AP = 0
    hit = 0
    for i, v in enumerate(ret):
        if v in ans:
            hit += 1
            AP += hit / (i + 1)

    return AP / len(ans)


def score(file='submit.csv'):
    submit = pd.read_csv(file)
    ans = pd.read_csv('data/queries/ans_train.csv')

    N = 10
    MAP = 0
    for i in range(N):
        sub_docs = submit.retrieved_docs[i].split()
        ans_docs = ans.retrieved_docs[i].split()

        AP = calc_AP(sub_docs, ans_docs)
        recall = len(set(sub_docs) & set(ans_docs))

        print ('#%d Recall: %d(%f), AP: %f' % (i + 1, recall, recall / len(ans_docs), AP))
        MAP += AP

    print ('MAP: %f' % (MAP / N))


class Timer(object):
    """ A quick tic-toc timer

    Credit: http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    """

    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.elapsed = None

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.tstart
        if self.verbose:
            if self.name:
                print ('[%s]' % self.name,)
            print ('Elapsed: %s' % self.elapsed)
