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
    Assumes a separate labels.csv file in the same directory for ground truth.
    """
    docs = []
    labels = []
    
    # Read documents from .txt files
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    for file_path in sorted(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            docs.append(f.read())
            
    # Read labels from labels.csv
    labels_file_path = os.path.join(directory, 'labels.csv')
    if not os.path.exists(labels_file_path):
        print(f"Error: '{labels_file_path}' not found. Cannot determine ground truth.")
        return docs, [], set()
        
    try:
        df = pd.read_csv(labels_file_path, header=0, names=['filename', 'label'])
        df['label'] = df['label'].str.strip().str.upper()
        if not all(df['label'].isin(['R', 'NR'])):
            raise ValueError("Labels must be 'R' or 'NR'.")
        
        # Match labels to document order
        labels_map = dict(zip(df['filename'], df['label']))
        for file_path in sorted(file_paths):
            filename = os.path.basename(file_path)
            labels.append(labels_map.get(filename, 'NR')) # Default to 'NR' if no label is found
            
        ground_truth = {i for i, label in enumerate(labels) if label == 'R'}
        return docs, labels, ground_truth
        
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return docs, [], set()

# --- Rest of the code from the combined script (no changes needed here) ---
def evaluate_query(retrieved, relevant):
    """
    Calculates precision, recall, and F1-score.
    :param retrieved: A set of retrieved document IDs.
    :param relevant: A set of ground-truth relevant document IDs.
    """
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def calculate_map(doc_scores, true_relevant):
    """
    Calculates Mean Average Precision (MAP) from ranked document scores.
    :param doc_scores: A list of (document_id, score) tuples, sorted by score.
    :param true_relevant: A set of ground-truth relevant document IDs.
    """
    average_precision = 0
    relevant_count = 0
    for i, (doc_id, _) in enumerate(doc_scores, 1):
        if doc_id in true_relevant:
            relevant_count += 1
            average_precision += relevant_count / i
    return average_precision / len(true_relevant) if relevant_count > 0 else 0

def jaccard_similarity(doc_terms, query_terms):
    """Calculates Jaccard similarity between two sets of terms."""
    doc_set = set(doc_terms)
    query_set = set(query_terms)
    intersection = len(doc_set & query_set)
    union = len(doc_set | query_set)
    return intersection / union if union > 0 else 0.0

def dice_similarity(doc_terms, query_terms):
    """Calculates Dice similarity between two sets of terms."""
    doc_set = set(doc_terms)
    query_set = set(query_terms)
    intersection = len(doc_set & query_terms)
    return (2 * intersection) / (len(doc_set) + len(query_set)) if (len(doc_set) + len(query_set)) > 0 else 0.0

# --- Retrieval Models ---

def run_boolean_retrieval(docs, processed_docs, query, true_relevant_indices):
    """Performs Boolean retrieval with an inverted index."""
    print("-----------------------------------")
    print("## Boolean Retrieval Model")
    print("-----------------------------------")

    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(processed_docs):
        for term in set(doc):  # Use set for unique terms per doc
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
    relevant_texts = [docs[i] for i in result_ids]
    metrics = evaluate_query(result_ids, true_relevant_indices)

    print(f"Query: '{query}'")
    print(f"Preprocessed Query (terms only): {preprocess(query)}")
    print(f"Inverted Index: {dict(inverted_index)}")
    print(f"Relevant Document IDs: {result_ids}")
    print(f"Relevant Documents:\n" + "\n".join(f"  - {text}" for text in relevant_texts))
    print(f"Evaluation Metrics: {metrics}\n")

def run_bim_retrieval(docs, processed_docs, true_relevant_indices):
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

    RSV_THRESHOLD = 0
    bim_relevant_indices = [j for j, score in enumerate(rsv_scores) if score > RSV_THRESHOLD]
    bim_relevant_docs = [f"d{i + 1}" for i in bim_relevant_indices]
    
    true_relevant_docs_str = {f"d{i + 1}" for i in true_relevant_indices}
    
    metrics = evaluate_query(bim_relevant_docs, true_relevant_docs_str)
    
    doc_scores = [(f"d{j+1}", score) for j, score in enumerate(rsv_scores)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    K = 2
    top_k = [doc for doc, _ in doc_scores[:K]]
    precision_at_k = len(set(top_k) & true_relevant_docs_str) / K if K > 0 else 0
    
    map_score = calculate_map(doc_scores, true_relevant_docs_str)

    print(f"Query: '{query}'")
    print(f"Preprocessed Query: {query_terms}")
    print(f"RSV Scores: {rsv_scores}")
    print(f"Relevant Documents (RSV > {RSV_THRESHOLD}): {bim_relevant_docs}")
    print(f"Evaluation Metrics: {metrics}")
    print(f"Precision@{K}: {precision_at_k:.4f}")
    print(f"MAP: {map_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar([f"d{i + 1}" for i in range(N)], rsv_scores, color='lightcoral')
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
    sns.heatmap(incidence_matrix, xticklabels=[f"d{i+1}" for i in range(N)], yticklabels=terms, cmap='Blues', annot=True)
    plt.title('Incidence Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

def run_vsm_retrieval(docs, processed_docs, true_relevant_indices):
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

    true_relevant_docs_str = {f"d{i + 1}" for i in true_relevant_indices}
    SIM_THRESHOLD = 0
    
    cosine_relevant_docs_str = [f"d{j+1}" for j, sim in enumerate(cosine_similarities) if sim > SIM_THRESHOLD]
    cosine_metrics = evaluate_query(cosine_relevant_docs_str, true_relevant_docs_str)
    
    cosine_doc_scores = sorted([(f"d{j+1}", s) for j, s in enumerate(cosine_similarities)], key=lambda x: x[1], reverse=True)
    K = 2
    cosine_precision_at_k = len(set([doc for doc, _ in cosine_doc_scores[:K]]) & true_relevant_docs_str) / K
    cosine_map = calculate_map(cosine_doc_scores, true_relevant_docs_str)

    print("### Cosine Similarity")
    print(f"Cosine Similarities: {[f'{s:.4f}' for s in cosine_similarities]}")
    print(f"Relevant Documents: {cosine_relevant_docs_str}")
    print(f"Metrics: {cosine_metrics}")
    print(f"Precision@{K}: {cosine_precision_at_k:.4f}")
    print(f"MAP: {cosine_map:.4f}\n")

    jaccard_relevant_docs_str = [f"d{j+1}" for j, sim in enumerate(jaccard_similarities) if sim > SIM_THRESHOLD]
    jaccard_metrics = evaluate_query(jaccard_relevant_docs_str, true_relevant_docs_str)
    
    jaccard_doc_scores = sorted([(f"d{j+1}", s) for j, s in enumerate(jaccard_similarities)], key=lambda x: x[1], reverse=True)
    jaccard_precision_at_k = len(set([doc for doc, _ in jaccard_doc_scores[:K]]) & true_relevant_docs_str) / K
    jaccard_map = calculate_map(jaccard_doc_scores, true_relevant_docs_str)

    print("### Jaccard Similarity")
    print(f"Jaccard Similarities: {[f'{s:.4f}' for s in jaccard_similarities]}")
    print(f"Relevant Documents: {jaccard_relevant_docs_str}")
    print(f"Metrics: {jaccard_metrics}")
    print(f"Precision@{K}: {jaccard_precision_at_k:.4f}")
    print(f"MAP: {jaccard_map:.4f}\n")

    dice_relevant_docs_str = [f"d{j+1}" for j, sim in enumerate(dice_similarities) if sim > SIM_THRESHOLD]
    dice_metrics = evaluate_query(dice_relevant_docs_str, true_relevant_docs_str)
    
    dice_doc_scores = sorted([(f"d{j+1}", s) for j, s in enumerate(dice_similarities)], key=lambda x: x[1], reverse=True)
    dice_precision_at_k = len(set([doc for doc, _ in dice_doc_scores[:K]]) & true_relevant_docs_str) / K
    dice_map = calculate_map(dice_doc_scores, true_relevant_docs_str)

    print("### Dice Similarity")
    print(f"Dice Similarities: {[f'{s:.4f}' for s in dice_similarities]}")
    print(f"Relevant Documents: {dice_relevant_docs_str}")
    print(f"Metrics: {dice_metrics}")
    print(f"Precision@{K}: {dice_precision_at_k:.4f}")
    print(f"MAP: {dice_map:.4f}\n")

    plt.figure(figsize=(10, 6))
    plt.bar([f"d{i+1}" for i in range(N)], cosine_similarities, color='skyblue')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Documents to Query')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar([f"d{i+1}" for i in range(N)], jaccard_similarities, color='lightgreen')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Jaccard Similarity')
    plt.title('Jaccard Similarity of Documents to Query')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar([f"d{i+1}" for i in range(N)], dice_similarities, color='lightcoral')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Dice Similarity')
    plt.title('Dice Similarity of Documents to Query')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(tfidf_docs, xticklabels=[f"d{i+1}" for i in range(N)], yticklabels=terms, cmap='YlGnBu', annot=True)
    plt.title('TF-IDF Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

def main():
    # --- Instructions for setup ---
    print("--- Setup Instructions ---")
    print("Please create a directory named 'docs' in the same folder as this script.")
    print("Place all your .txt documents inside the 'docs' directory.")
    print("Also, create a 'labels.csv' file inside 'docs' with the following format:")
    print("filename,label")
    print("doc1.txt,R")
    print("doc2.txt,NR")
    print("...etc.")
    print("where 'R' indicates a relevant document and 'NR' is non-relevant.")
    print("----------------------------\n")

    docs_directory = "docs"
    query = "information AND retrieval" # Example query for Boolean model
    
    docs, labels, ground_truth_indices = read_documents(docs_directory)
    if not docs:
        print("No documents found. Exiting.")
        return

    processed_docs = [preprocess(d) for d in docs]
    
    print("--- Document and Preprocessing Info ---")
    print("Documents:", [doc[:50] + "..." for doc in docs])
    print("Labels:", labels)
    print("Processed Documents:", [doc[:5] for doc in processed_docs])
    print("Ground Truth Relevant Document Indices:", ground_truth_indices)
    print("\n" * 2)

    run_boolean_retrieval(docs, processed_docs, query, ground_truth_indices)
    print("\n" * 2)
    run_bim_retrieval(docs, processed_docs, ground_truth_indices)
    print("\n" * 2)
    run_vsm_retrieval(docs, processed_docs, ground_truth_indices)

if __name__ == "__main__":
    main()