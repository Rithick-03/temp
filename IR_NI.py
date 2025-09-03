import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Documents and Query ---
docs = [
    "information requirement query considers user feedback",
    "information retrieval query depends on retrieval model",
    "prediction problem many problems in retrieval as prediction",
    "search engine one application of retrieval models",
    "feedback improves query prediction"
]
query = "feedback improves query prediction"

# --- Step 2: Build term-document frequency matrix ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs).toarray()
terms = vectorizer.get_feature_names_out()
q_vec = vectorizer.transform([query]).toarray()

N = len(docs)  # total docs
df = np.sum(X > 0, axis=0)  # number of docs each term appears in
print("df",df)
# --- Stage 1: BIM without relevance info ---
trk = np.log((N - df) / (df + 1e-10))  # weights

# Weighting
X_weighted = X * trk
q_weighted = q_vec * trk

# Cosine similarity
similarities = cosine_similarity(q_weighted, X_weighted)[0]
ranked_indices = np.argsort(-similarities)

print("Stage 1 Ranking (no relevance info):")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"Rank {rank}: D{idx+1} (score={similarities[idx]:.3f})")

# --- Stage 2: Apply BIM with relevance info on top-2 ranked docs ---
top_docs_idx = ranked_indices[:2]  # take top 2 documents
R = len(top_docs_idx)              # number of relevant docs considered

# Count rk = number of relevant docs containing term
rk = np.sum(X[top_docs_idx, :] > 0, axis=0)

# Formula: w_k = log( (rk + 0.5) / (R - rk + 0.5) ) - log( (df - rk + 0.5) / (N - df - R + rk + 0.5) )
wk = np.log((rk + 0.5) / (R - rk + 0.5)) - np.log((df - rk + 0.5) / (N - df - R + rk + 0.5))

# Reweight docs and query
X_rel_weighted = X * wk
q_rel_weighted = q_vec * wk

# Cosine similarity again
similarities_rel = cosine_similarity(q_rel_weighted, X_rel_weighted)[0]
ranked_indices_rel = np.argsort(-similarities_rel)

print("\nStage 2 Ranking (with relevance info from top-2 docs):")
for rank, idx in enumerate(ranked_indices_rel, 1):
    print(f"Rank {rank}: D{idx+1} (score={similarities_rel[idx]:.3f})")
# ------------------

import os
import math
import pandas as pd
import re

# -------------------------------
# 1. Preprocessing function
# -------------------------------
def preprocess(text):
    stop_words = {"this", "is", "a", "of", "with", "always", "the", "here", "yet", "about", "and"}
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters
    tokens = [t for t in text.split() if t not in stop_words]
    return tokens

# -------------------------------
# 2. Load documents
# -------------------------------
data_folder = "your_text_files_folder"
docs = {}
for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r") as f:
            docs[filename] = preprocess(f.read())

print("Documents loaded and tokenized:\n")
for name, tokens in docs.items():
    print(name, ":", tokens)

# -------------------------------
# 3. Build Term-Document Matrix
# -------------------------------
# Collect all terms
all_terms = sorted(set(term for tokens in docs.values() for term in tokens))

# Initialize matrix
td_matrix = pd.DataFrame(0, index=all_terms, columns=docs.keys())

# Fill counts
for doc, tokens in docs.items():
    for term in tokens:
        td_matrix.loc[term, doc] += 1

print("\nTerm-Document Matrix:")
print(td_matrix)

# -------------------------------
# 4. Compute IDF
# -------------------------------
N = len(docs)
idf = {}
for term in td_matrix.index:
    df_count = (td_matrix.loc[term] > 0).sum()
    idf[term] = math.log10(N / df_count) if df_count > 0 else 0

idf_df = pd.DataFrame.from_dict(idf, orient='index', columns=['idf'])

# -------------------------------
# 5. Compute TF-IDF
# -------------------------------
tfidf = td_matrix.T * idf_df["idf"].values
print("\nTF-IDF Matrix:")
print(tfidf)

# -------------------------------
# 6. Cosine similarity with query
# -------------------------------
def dot_product(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def magnitude(v):
    return sum(x*x for x in v) ** 0.5

def cosine_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product(v1, v2) / (mag1 * mag2)

# Example query
query = "fun information retrieval"
query_tokens = preprocess(query)

# Build query vector in same term space
query_vec = []
for term in tfidf.columns:
    tf = query_tokens.count(term)
    q_weight = tf * idf.get(term, 0)
    query_vec.append(q_weight)

# Compute similarities
scores = []
for doc in tfidf.index:
    row_vec = tfidf.loc[doc].tolist()
    sim = cosine_similarity(query_vec, row_vec)
    scores.append((doc, sim))

scores.sort(key=lambda x: x[1], reverse=True)

print("\nRanking of documents for query:", query)
for doc, sim in scores:
    print(f"{doc}: {sim:.4f}")

import numpy as np
import re

# ----------------------------
# Step 1: Documents and Query
# ----------------------------
docs = [
    "information requirement query considers user feedback",
    "information retrieval query depends on retrieval model",
    "prediction problem many problems in retrieval as prediction",
    "search engine one application of retrieval models",
    "feedback improves query prediction"
]
query = "feedback improves query prediction"

# ----------------------------
# Step 2: Preprocessing
# ----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    return text.split()

tokenized_docs = [tokenize(d) for d in docs]
query_tokens = tokenize(query)

# Build vocabulary
vocab = sorted(set(term for doc in tokenized_docs for term in doc))
term_index = {term: i for i, term in enumerate(vocab)}

# ----------------------------
# Step 3: Term-Document Matrix
# ----------------------------
N = len(docs)
V = len(vocab)

X = np.zeros((N, V), dtype=int)  # term-document frequency matrix
for i, doc in enumerate(tokenized_docs):
    for term in doc:
        X[i, term_index[term]] += 1
print(X)

# Query vector
q_vec = np.zeros(V, dtype=int)
for term in query_tokens:
    if term in term_index:
        q_vec[term_index[term]] += 1
print(q_vec)

# ----------------------------
# Stage 1: BIM without relevance info
# ----------------------------
df = np.sum(X > 0, axis=0)        # document frequency
trk = np.log((N - df + 1e-10) / (df + 1e-10))  # BIM weights
print(df)
print(trk)

# Weight docs and query
X_weighted = X * trk
q_weighted = q_vec * trk

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    mag = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
    return 0 if mag == 0 else dot / mag

# Compute similarities
similarities = np.array([cosine_similarity(q_weighted, row) for row in X_weighted])
ranked_indices = np.argsort(-similarities)

print("Stage 1 Ranking (no relevance info):")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"Rank {rank}: D{idx+1} (score={similarities[idx]:.3f})")

# ----------------------------
# Stage 2: BIM with relevance info
# ----------------------------
top_docs_idx = ranked_indices[:2]   # assume top-2 relevant
R = len(top_docs_idx)
print(R)

# rk = number of relevant docs containing term
rk = np.sum(X[top_docs_idx, :] > 0, axis=0)
print(rk)


# BIM relevance feedback weighting formula
print(rk,R,N,df)
wk = np.log((rk + 0.5) / (R - rk + 0.5)) - np.log((df - rk + 0.5) / (N - df - R + rk + 0.5))
print(wk)

# Reweight docs and query
X_rel_weighted = X * wk
q_rel_weighted = q_vec * wk

# Compute similarities again
similarities_rel = np.array([cosine_similarity(q_rel_weighted, row) for row in X_rel_weighted])
ranked_indices_rel = np.argsort(-similarities_rel)

print("\nStage 2 Ranking (with relevance info from top-2 docs):")
for rank, idx in enumerate(ranked_indices_rel, 1):
    print(f"Rank {rank}: D{idx+1} (score={similarities_rel[idx]:.3f})")
