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
