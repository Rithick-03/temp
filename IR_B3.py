import pandas as pd
import numpy as np
from math import log
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# --- Custom Setup without NLTK ---
stop_words = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can't", "cannot", "could"
])

# Operators for Boolean queries
operators = {"and", "or", "not"}

def custom_tokenize(text):
    """Splits text into words using a simple regex, removing punctuation."""
    return re.findall(r'\b\w+\b', text.lower())

def custom_stem(word):
    """A very simple Porter-like stemming implementation."""
    if len(word) > 2:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("s"):
            return word[:-1]
    return word

# --- Preprocessing Function ---
def preprocess(text):
    """Preprocesses a text string by tokenizing, stemming, and removing stopwords."""
    tokens = custom_tokenize(text)
    tokens = [custom_stem(t) for t in tokens if t not in stop_words and t not in operators]
    return tokens

# --- Updated Document Reading Function ---
def read_documents(directory):
    """
    Reads documents from multiple .txt files in a given directory.
    This version does not expect a labels file.
    """
    docs = []
    file_names = []
    
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    for file_path in sorted(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs.append(f.read())
                file_names.append(os.path.basename(file_path))
        except IOError:
            print(f"Error reading file: {file_path}")
            continue
            
    return docs, file_names

# --- Retrieval Models ---

def run_boolean_retrieval(docs, processed_docs, query, file_names):
    """Performs Boolean retrieval with an inverted index."""
    print("-----------------------------------")
    print("## Boolean Retrieval Model")
    print("-----------------------------------")

    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(processed_docs):
        for term in set(doc):
            inverted_index[term].append(doc_id)

    def process_query(query_str, inverted_index, num_docs):
        tokens = custom_tokenize(query_str)
        result = None
        operator = None
        all_docs = set(range(num_docs))
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in operators:
                operator = token
                i += 1
                continue
            term = custom_stem(token)
            current_docs = set(inverted_index.get(term, []))

            if operator == "not":
                current_docs = all_docs - current_docs
                operator = None
            
            if result is None:
                result = current_docs
            elif operator == "and":
                result = result.intersection(current_docs)
                operator = None
            elif operator == "or":
                result = result.union(current_docs)
                operator = None
            i += 1
        return result if result is not None else set()

    result_ids = process_query(query, inverted_index, len(docs))
    relevant_files = [file_names[i] for i in result_ids]

    print(f"Query: '{query}'")
    print(f"Preprocessed Query (terms only): {preprocess(query)}")
    print(f"Inverted Index: {dict(inverted_index)}")
    print(f"Retrieved Document Files: {relevant_files}\n")

def run_bim_retrieval(docs, processed_docs, file_names):
    """Performs retrieval using the Binary Independence Model (BIM)."""
    print("-----------------------------------")
    print("## Binary Independence Model (BIM)")
    print("-----------------------------------")

    N = len(docs)
    terms = sorted(set(t for doc in processed_docs for t in doc))
    df_dict = {term: sum(1 for doc in processed_docs if term in doc) for term in terms}
    bim_idf = {term: log((N - df_dict[term] + 0.5) / (df_dict[term] + 0.5)) for term in terms}

    query = 'information retrieval'
    query_terms = preprocess(query)
    
    rsv_scores = []
    for j in range(N):
        score = sum(bim_idf.get(q, 0) for q in query_terms if q in processed_docs[j])
        rsv_scores.append(score)

    doc_scores = sorted(zip(file_names, rsv_scores), key=lambda x: x[1], reverse=True)

    print(f"Query: '{query}'")
    print(f"Preprocessed Query: {query_terms}")
    print("RSV Score Ranking:")
    for file, score in doc_scores:
        print(f"  - {file}: {score:.4f}")

    RSV_THRESHOLD = 0
    bim_relevant_files = [file for file, score in doc_scores if score > RSV_THRESHOLD]
    print(f"\nRetrieved Documents (RSV > {RSV_THRESHOLD}): {bim_relevant_files}\n")

    plt.figure(figsize=(10, 6))
    plt.bar(file_names, rsv_scores, color='lightcoral')
    plt.axhline(y=RSV_THRESHOLD, color='b', linestyle='--', label=f'Threshold ({RSV_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('RSV Score')
    plt.title('BIM RSV Scores for Documents')
    plt.legend()
    plt.show()
    
    incidence_matrix = np.zeros((len(terms), N), dtype=int)
    for i, term in enumerate(terms):
        for j, doc in enumerate(processed_docs):
            if term in doc:
                incidence_matrix[i, j] = 1

    plt.figure(figsize=(12, 8))
    sns.heatmap(incidence_matrix, xticklabels=file_names, yticklabels=terms, cmap='Blues', annot=True)
    plt.title('Incidence Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

def run_vsm_retrieval(docs, processed_docs, file_names):
    """Performs retrieval using the Vector Space Model (VSM) with different similarities."""
    print("-----------------------------------")
    print("## Vector Space Model (VSM)")
    print("-----------------------------------")

    N = len(docs)
    terms = sorted(set(t for doc in processed_docs for t in doc))
    
    query = 'information retrieval'
    processed_query = preprocess(query)

    tfidf_docs = np.zeros((len(terms), N))
    for i, term in enumerate(terms):
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        for j, doc in enumerate(processed_docs):
            tf = doc.count(term)
            tfidf_docs[i, j] = tf * idf

    tfidf_query = np.zeros(len(terms))
    for i, term in enumerate(terms):
        tf = processed_query.count(term)
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        tfidf_query[i] = tf * idf

    cosine_similarities = []
    for j in range(N):
        doc_vec = tfidf_docs[:, j]
        if np.linalg.norm(doc_vec) > 0 and np.linalg.norm(tfidf_query) > 0:
            cos = np.dot(doc_vec, tfidf_query) / (np.linalg.norm(doc_vec) * np.linalg.norm(tfidf_query))
        else:
            cos = 0
        cosine_similarities.append(cos)

    jaccard_similarities = [jaccard_similarity(doc, processed_query) for doc in processed_docs]
    dice_similarities = [dice_similarity(doc, processed_query) for doc in processed_docs]

    print("### Cosine Similarity Ranking")
    cosine_doc_scores = sorted(zip(file_names, cosine_similarities), key=lambda x: x[1], reverse=True)
    for file, score in cosine_doc_scores:
        print(f"  - {file}: {score:.4f}")

    print("\n### Jaccard Similarity Ranking")
    jaccard_doc_scores = sorted(zip(file_names, jaccard_similarities), key=lambda x: x[1], reverse=True)
    for file, score in jaccard_doc_scores:
        print(f"  - {file}: {score:.4f}")

    print("\n### Dice Similarity Ranking")
    dice_doc_scores = sorted(zip(file_names, dice_similarities), key=lambda x: x[1], reverse=True)
    for file, score in dice_doc_scores:
        print(f"  - {file}: {score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(file_names, cosine_similarities, color='skyblue')
    plt.xlabel('Documents')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Documents to Query')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(file_names, jaccard_similarities, color='lightgreen')
    plt.xlabel('Documents')
    plt.ylabel('Jaccard Similarity')
    plt.title('Jaccard Similarity of Documents to Query')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(file_names, dice_similarities, color='lightcoral')
    plt.xlabel('Documents')
    plt.ylabel('Dice Similarity')
    plt.title('Dice Similarity of Documents to Query')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(tfidf_docs, xticklabels=file_names, yticklabels=terms, cmap='YlGnBu', annot=True)
    plt.title('TF-IDF Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

def main():
    print("--- Setup Instructions ---")
    print("Please create a directory named 'docs' in the same folder as this script.")
    print("Place all your .txt documents inside the 'docs' directory.")
    print("----------------------------\n")

    docs_directory = "docs"
    query = "information AND retrieval"
    
    docs, file_names = read_documents(docs_directory)
    if not docs:
        print("No documents found. Exiting.")
        return

    processed_docs = [preprocess(d) for d in docs]
    
    print("--- Document and Preprocessing Info ---")
    print("File Names:", file_names)
    print("Number of Documents:", len(docs))
    print("Example Processed Document (first 5 terms):", processed_docs[0][:5])
    print("\n" * 2)

    run_boolean_retrieval(docs, processed_docs, query, file_names)
    print("\n" * 2)
    run_bim_retrieval(docs, processed_docs, file_names)
    print("\n" * 2)
    run_vsm_retrieval(docs, processed_docs, file_names)

if __name__ == "__main__":
    main()
