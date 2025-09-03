import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import re

# --- Custom Functions to replace NLTK ---

def custom_word_tokenize(text):
    """
    Custom tokenizer to split a text into words.
    Removes non-alphanumeric characters and converts to lowercase.
    """
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def custom_porter_stemmer(word):
    """
    A very basic custom stemmer approximating the Porter Stemmer's rules.
    This is not a full implementation but handles common cases.
    """
    if len(word) > 2:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("s"):
            return word[:-1]
    return word

# --- Data and main logic ---

database = [
    {"title": "Information requirement", "content": "query considers the user feedback as information requirement to search"},
    {"title": "Information retrieval", "content": "query depends on the model of information retrieval used"},
    {"title": "Prediction problem", "content": "Many problems in information retrieval can be viewed as prediction problems"},
    {"title": "Search", "content": "A search engine is one of applications of information retrieval models"}
]

new_docs = [
    {"title": "Feedback", "content": "feedback is typically used by the system to modify the query and improve prediction"},
    {"title": "information retrieval", "content": "ranking in information retrieval algorithms depends on user query"}
]

def binary_distance(title1, title2):
    return 0 if title1.lower() == title2.lower() else 1

def compute_cosine_similarity(docs, new_doc_content, threshold=0.85):
    vectorizer = TfidfVectorizer(stop_words='english')
    all_docs = [doc["content"] for doc in docs] + [new_doc_content]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    max_similarity = max(similarities) if similarities.size > 0 else 0
    is_duplicate = max_similarity > threshold
    return is_duplicate, max_similarity

def get_shingles(text, k=3):
    tokens = [custom_porter_stemmer(word) for word in custom_word_tokenize(text)]
    if len(tokens) < k:
        return set()
    return set(' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1))

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_jaccard_similarity(docs, new_doc_content, k=3, threshold=0.85):
    new_shingles = get_shingles(new_doc_content, k)
    max_similarity = 0
    for doc in docs:
        doc_shingles = get_shingles(doc["content"], k)
        similarity = jaccard_similarity(new_shingles, doc_shingles)
        max_similarity = max(max_similarity, similarity)
    is_duplicate = max_similarity > threshold
    return is_duplicate, max_similarity

def compute_bm25(docs, new_doc_content, k1=1.5, b=0.75, threshold=0.85):
    all_docs = [doc["content"] for doc in docs] + [new_doc_content]
    tokenized_docs = [[custom_porter_stemmer(word) for word in custom_word_tokenize(doc)] for doc in all_docs]

    doc_freq = Counter()
    for doc in tokenized_docs[:-1]:
        doc_freq.update(set(doc))
    
    avg_doc_len = sum(len(doc) for doc in tokenized_docs[:-1]) / len(docs)
    N = len(docs)
    scores = []
    
    query_terms = Counter(tokenized_docs[-1])

    for i, doc in enumerate(tokenized_docs[:-1]):
        doc_len = len(doc)
        score = 0
        for term, query_count in query_terms.items():
            if term in doc_freq:
                tf = doc.count(term)
                idf = math.log((N - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        scores.append(score)

    max_score = max(scores) if scores else 0
    normalized_score = max_score / (max_score + 1) if max_score > 0 else 0
    is_duplicate = normalized_score > threshold
    return is_duplicate, normalized_score

def plagiarism_checker(database, new_doc):
    # Binary Distance check for exact title match
    for db_doc in database:
        if binary_distance(new_doc["title"], db_doc["title"]) == 0:
            return {
                "title": new_doc["title"],
                "is_duplicate": True,
                "reason": "Exact title match",
                "similarity_score": 1.0,
                "method": "Binary Distance"
            }

    # Cosine Similarity check
    is_duplicate_cosine, cosine_sim = compute_cosine_similarity(database, new_doc["content"])
    if is_duplicate_cosine:
        return {
            "title": new_doc["title"],
            "is_duplicate": True,
            "reason": f"Cosine similarity {cosine_sim:.3f} > 0.85",
            "similarity_score": cosine_sim,
            "method": "Cosine Similarity"
        }

    # Jaccard Similarity check
    is_duplicate_jaccard, jaccard_sim = compute_jaccard_similarity(database, new_doc["content"])
    if is_duplicate_jaccard:
        return {
            "title": new_doc["title"],
            "is_duplicate": True,
            "reason": f"Jaccard similarity {jaccard_sim:.3f} > 0.85",
            "similarity_score": jaccard_sim,
            "method": "Jaccard Similarity"
        }

    # Okapi BM25 check
    is_duplicate_bm25, bm25_score = compute_bm25(database, new_doc["content"])
    if is_duplicate_bm25:
        return {
            "title": new_doc["title"],
            "is_duplicate": True,
            "reason": f"BM25 score {bm25_score:.3f} > 0.85",
            "similarity_score": bm25_score,
            "method": "Okapi BM25"
        }

    return {
        "title": new_doc["title"],
        "is_duplicate": False,
        "reason": "No significant similarity detected",
        "similarity_score": max(cosine_sim, jaccard_sim, bm25_score),
        "method": "All methods"
    }

# Main execution loop
results = []
for new_doc in new_docs:
    result = plagiarism_checker(database, new_doc)
    results.append(result)

# Display results
df = pd.DataFrame(results, columns=["title", "is_duplicate", "reason", "similarity_score", "method"])
print(df.to_string(index=False))

# Update database with non-duplicate documents
for new_doc, result in zip(new_docs, results):
    if not result["is_duplicate"]:
        database.append(new_doc)

print("\nUpdated Database:")
for doc in database:
    print(f"Title: {doc['title']}, Content: {doc['content']}")