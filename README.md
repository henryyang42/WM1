# A Small Information Retrieval System Using Vector Space Model with Rocchio Relevance Feedback

Web Mining Programming Homework 1

[https://inclass.kaggle.com/c/wm-2017-vsm-model](https://inclass.kaggle.com/c/wm-2017-vsm-model)

---

## 0 Introduction
In this programming homework we implement a small information retrival system using vector space model (VSM) with Rocchio relevance feedback (pseudo version) on CIBR010 dataset. We need to retrieve articles from around 47000 Chinese news articles and evaluate the performance using MAP@100 metric.

## 1 Methodology
### 1.1 Preprocessing
For our convenience, we transform the original corpus and queries into two ``.csv`` files and treat all texts  with Chinese segmentation using ``jieba``. The whole process takes about 10 minutes and the preprocessed files could be read into memory in a matter of seconds.

### 1.2 Vector Space Model
We create ``tfidfvectorizer`` which convert the corpus into document-term matrix. We use ``jieba``'s segmentation result to identify tokens. By maintaining a term to id dictionary, we could create ``(doc_i, term_j, raw_tf)`` tuples while recording inverse document frequency (idf) for the terms that appear in a doc. The ``(doc_i, term_j, raw_tf)`` tuples are later transformed to ``(doc_i, term_j, tfidf_weight)`` using configurable tfidf metric. The produced tuples can be made into a sparse matrix using doc_i as row, term_j as column and tfidf_weight as element. The ~11M tuples, at last, are fit into a sparse matrix of shape (46972, 670348).

The relevancy of a query and documents could be derived from performing cosine similarity measure with query vector and document matrix. We choose documents with top 100 highest scores for evaluation.

### 1.3 Rocchio Relevance Feedback
Since we don't know the exact relevant set for each query, we implement the pseudo version of Rocchio relevance feedback by assuming top results relevant.  Here, we define a threshold ``r_threshold``, the similarity our assuming relevant set should exceed, ``r_max`` the upper limit of documents as relevant set. With the help of ``scipy`` and ``numpy``, such idea could be formulate into 2 lines given computed similarities (``sims``):

```py
topN = min(len(sims[sims > r_treshold]), r_max)
query_vector = query_vector + r_beta / topN * np.sum(vectors[top100[:topN]], axis=0)
```
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/61d961eee905b400a73875d212a84ee76e644f79)

For tuning convenience, we take ``a=1`` and ``c=0`` from experience, leaving ``b`` as tuning parameter.

## 2 Experiments

### 2.1 Weighting 

| Method (TF/DF) | MAP@100 (train w/o rf) | MAP@100 (train w/ rf) |
| ------:| -----------:|-----------:|
| **natural/no** | 0.6903 | 0.4584 |
| **logarithm/idf** | 0.7936 | 0.8024 |
| **okapi** | 0.7703 | 0.7927 |

Different weighting schema with or without Rocchio relevance feedback.

---

### 2.2 Tuning Rocchio Relevance Feedback
![](https://i.imgur.com/liy4fPf.jpg =550x)

Performance under various weighting schema and beta value.

## 3 Discussions
In the experiments we compare different weighting schema and beta values. Performances of ``logarithm/idf`` and ``okapi`` is steadily high, while ``natural/no`` become lower and lower when beta increases. Maybe the top results in ``natural/no`` are not so relevant that mislead the query vector to the wrong direction. 