import os, re, csv, math
import numpy as np
from collections import Counter

# ---------- Preprocessing ----------
def preprocess(text):
    return re.findall(r'\w+', text.lower())

# ---------- Loading ----------
def load_txt(path):
    docs = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line != '':
                docs.append(line)
    return docs

def load_csv(path, column):
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            text = (row.get(column) or '').strip()
            if text:
                docs.append(text)
    return docs

def load_and_preprocess_docs(file_paths):
    docs = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            if text:
                docs.append(text)   # ⬅️ no preprocess here
    return docs


# ---------- Inverted Index ----------
def build_inverted_index(docs):
    index = {}
    for doc_id, text in enumerate(docs):          # each document has an ID
        for term in set(preprocess(text)):        # preprocess + unique terms
            index.setdefault(term, []).append(doc_id)
    return {t: sorted(set(ids)) for t, ids in index.items()}

def get_postings(inv, term):
    return set(inv.get(term.lower(), []))

def boolean_and(inv, t1, t2):
    return sorted(get_postings(inv, t1) & get_postings(inv, t2))

def boolean_or(inv, t1, t2):
    return sorted(get_postings(inv, t1) | get_postings(inv, t2))

def boolean_not(inv, t, N):
    return sorted(set(range(N)) - get_postings(inv, t))

def eval_query(query, inv, N):
    tokens = query.lower().split()
    result = set(range(N))   # start with all docs

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == "and":
            i += 1
            result &= get_postings(inv, tokens[i])

        elif token == "or":
            i += 1
            result |= get_postings(inv, tokens[i])

        elif token == "not":
            i += 1
            result -= get_postings(inv, tokens[i])

        else:
            # first term (before any AND/OR/NOT)
            result &= get_postings(inv, token)

        i += 1

    return sorted(result)


# ---------- Vector Space Model ----------
def build_tfidf(docs):
    tokens = [preprocess(d) for d in docs]
    N = len(tokens)
    vocab = {t: i for i, t in enumerate(set(t for d in tokens for t in d))}
    V = len(vocab)
    tf = np.zeros((V, N))
    for j, d in enumerate(tokens):
        for t, c in Counter(d).items():
            tf[vocab[t], j] = c
    df = np.count_nonzero(tf > 0, axis=1)
    idf = np.log((N + 1) / (df + 1)) + 1
    tfidf = (tf.T * idf).T
    return vocab, tf, tfidf, idf

def vectorize_query(query, vocab, idf):
    q = np.zeros(len(vocab))
    for t, c in Counter(preprocess(query)).items():
        if t in vocab:
            q[vocab[t]] = c
    return q * idf

def cosine_sim(q, D):
    return (q @ D) / (np.linalg.norm(q) + 1e-12) / (np.linalg.norm(D, axis=0) + 1e-12)

def jaccard_sim(q, D):
    qb = (q > 0).astype(int)[:, None]
    Db = (D > 0).astype(int)
    inter = (qb * Db).sum(axis=0)
    union = (qb + Db - (qb * Db)).sum(axis=0) + 1e-12
    return inter / union

def dice_sim(q, D):
    qb = (q > 0).astype(int)[:, None]
    Db = (D > 0).astype(int)
    inter = (qb * Db).sum(axis=0)
    denom = qb.sum() + Db.sum(axis=0) + 1e-12
    return 2 * inter / denom

# ---------- BIM ----------
import numpy as np
from collections import Counter

def bim_phase1(tf, vocab, docs, query, smooth=0.5):
    N = len(docs)
    qterms = preprocess(query)

    dk = np.array([np.count_nonzero(tf[vocab[t]]) if t in vocab else 0 for t in qterms], float)
    pk = np.full_like(dk, 0.5)   # assume uniform prior relevance
    qk = (dk + smooth) / (N + 2 * smooth)

    w = np.log((pk * (1 - qk) + 1e-12) / (qk * (1 - pk) + 1e-12))
    return qterms, w


def bim_phase2(tf, vocab, docs, query, rel_set, smooth=0.5):
    N = len(docs)
    Nr = len(rel_set)
    qterms = preprocess(query)

    dk = np.array([np.count_nonzero(tf[vocab[t]]) if t in vocab else 0 for t in qterms], float)
    rk = np.zeros_like(dk)

    for i, t in enumerate(qterms):
        if t in vocab:
            # count how many relevant docs contain term t
            rk[i] = sum(1 for d in rel_set if tf[vocab[t], d] > 0)

    pk = (rk + smooth) / (Nr + 2 * smooth)
    qk = (dk - rk + smooth) / ((N - Nr) + 2 * smooth)

    w = np.log((pk * (1 - qk) + 1e-12) / (qk * (1 - pk) + 1e-12))
    return qterms, w


def bim_score(tf, vocab, qterms, w):
    N = tf.shape[1]
    scores = np.zeros(N)
    for j in range(N):  # each document
        s = 0
        for i, t in enumerate(qterms):
            if t in vocab and tf[vocab[t], j] > 0:
                s += w[i]
        scores[j] = s
    return scores


def bim_search(tf, vocab, docs, query, labels=None, iters=3, top_rel=2):
    if labels is None:
        qterms, w = bim_phase1(tf, vocab, docs, query)
        scores = bim_score(tf, vocab, qterms, w)
        rel_set = set(np.argsort(-scores)[:top_rel])
    else:
        rel_set = set(i for i, l in enumerate(labels) if l == 1)
        qterms, w = bim_phase2(tf, vocab, docs, query, rel_set)

    for _ in range(iters):
        scores = bim_score(tf, vocab, qterms, w)
        rel_set = set(np.argsort(-scores)[:top_rel])
        qterms, w = bim_phase2(tf, vocab, docs, query, rel_set)

    final_order = np.argsort(-scores)
    return final_order, rel_set, scores[final_order]

# ---------- Example ----------
if __name__ == "__main__":
    docs = [
        "The cat sat on the mat",
        "The dog chased the cat",
        "Dogs and cats are friends"
    ]
    #document_paths = ['d1.txt', "d2.txt", "d3.txt"]
    #docs = load_and_preprocess_docs(document_paths)
    print(docs)
    inv = build_inverted_index(docs)
    N = len(docs)
    print("AND:", boolean_and(inv, "cats", "dogs"))
    print("OR :", boolean_or(inv, "cats", "dogs"))
    print("NOT:", boolean_not(inv, "birds", len(docs)))
    q1 = "cats AND dogs NOT birds"
    print(q1, "->", eval_query(q1, inv, N))

    vocab, tf, tfidf, idf = build_tfidf(docs)
    qvec = vectorize_query("cats dogs", vocab, idf)
    print("Cosine :", cosine_sim(qvec, tfidf))
    print("Jaccard:", jaccard_sim(qvec, tfidf))
    print("Dice   :", dice_sim(qvec, tfidf))

    order, rel_set, scores = bim_search(tf, vocab, docs, "cats dogs", iters=2, top_rel=2)
    print("BIM ranking:", order, "Relevant set:", rel_set)
