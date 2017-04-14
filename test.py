import argparse
from utils import *
from tfidfvectorizer import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', dest='ntcir_dir', default='data/CIRB010', help='The directiry of NTCIR documents.')
parser.add_argument('-i', dest='query_file', default='data/queries/query-train.xml', help='The input query file.')
parser.add_argument('-o', dest='output_file', default='submit.csv', help='The output ranked list file.')
parser.add_argument('-m', dest='model_dir', default='data/model', help='The input model directory.')
parser.add_argument('-r', dest='revelance', help='Revelance feedback.', action='store_true')
args = parser.parse_args()

# Tuning parameters
r_treshold = 0.2
r_N = 3
r_beta = 0.55
r_maxN = 10

if __name__ == "__main__":
    print ("Starting with \nrevelance = %s\nr_treshold = %f\nr_N = %d\nr_beta = %f\nr_maxN = %d" % (args.revelance, r_treshold, r_N, r_beta, r_maxN))
    with Timer('Generating docs...'):
        list_dirs = os.walk(args.ntcir_dir)
        dict_list = []
        for root, dirs, files in list_dirs:
            if not dirs:
                for file in files:
                    dict_list.append(doc2dict(os.path.join(root, file)))

        doc = pd.DataFrame(dict_list)
        doc.to_csv('doc.csv')

    with Timer('Generating queries...'):
        query = pd.DataFrame(query2dicts(args.query_file))
        query.to_csv('query.csv')
        n_query = len(query)

    with Timer('Making corpus...'):
        corpus = make_corpus(doc, query)

    with Timer('Vectorizing...'):
        vectorizer = tfidfvectorizer()
        vectors = vectorizer.vectorize(corpus)

    with Timer('Ranking...'):
        with open(args.output_file, 'w') as f_csv:
            print('query_id,retrieved_docs', file=f_csv)
            for i in range(n_query):
                query_vector = vectors[-n_query + i]

                dists = cosine_similarity(vectors, query_vector).T[0]
                top100 = dists.argsort()[::-1][1:101]
                if args.revelance:
                    for _ in range(r_N):
                        topN = min(len(dists[dists > r_treshold]), r_maxN)

                        query_vector = query_vector + r_beta / topN * np.sum(vectors[top100[:topN]], axis=0)
                        dists = cosine_similarity(vectors, query_vector).T[0]
                        top100 = dists.argsort()[::-1][1:101]

                top100 = list(doc.id[top100].fillna(' ').as_matrix())
                print('%s' % (query.number[i][-3:]), end=",", file=f_csv)
                print(" ".join(top100), file=f_csv)
