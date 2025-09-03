import os
import re
import numpy as np

def load_documents(directory):
    
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append((filename, file.read()))
    return documents

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = text.split()
    
    stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'and', 'or', 'to', 'for', 'of'}
    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(word)
    
    return clean_tokens


def build_inverted_index(processed_documents):

    inverted_index = []
    
    for doc_id, tokens in processed_documents:
        doc_term_counts = []
        for token in tokens:
            found = False
            for i in range(len(doc_term_counts)):
                if doc_term_counts[i][0] == token:
                    doc_term_counts[i][1] += 1
                    found = True
                    break
            if not found:
                doc_term_counts.append([token, 1])
        
        for term_tf in doc_term_counts:
            term = term_tf[0]
            tf = term_tf[1]
            
            term_found_in_index = False
            for i in range(len(inverted_index)):
                if inverted_index[i][0] == term:
                    inverted_index[i].append([doc_id, tf])
                    term_found_in_index = True
                    break
            
            if not term_found_in_index:
                inverted_index.append([term, [doc_id, tf]])

    return inverted_index

def find_term_postings(term, inverted_index):
    for entry in inverted_index:
        if entry[0] == term:
            return [post[0] for post in entry[1:]]
    return []

def boolean_retrieval_extended(query, inverted_index, all_doc_ids_set):

    query_parts = query.lower().split()
    
    current_result = set()
    
    i = 0
    if query_parts[0] == "not":
        if len(query_parts) > 1:
            term_docs = set(find_term_postings(query_parts[1], inverted_index))
            current_result = all_doc_ids_set.difference(term_docs)
            i = 2
    else:
        current_result = set(find_term_postings(query_parts[0], inverted_index))
        i = 1
    
    while i < len(query_parts):
        operator = query_parts[i]
        
        if operator == "and":
            term_docs = set(find_term_postings(query_parts[i+1], inverted_index))
            current_result.intersection_update(term_docs)
            i += 2
        elif operator == "or":
            term_docs = set(find_term_postings(query_parts[i+1], inverted_index))
            current_result.update(term_docs)
            i += 2
        elif operator == "not":
            term_docs = set(find_term_postings(query_parts[i+1], inverted_index))
            current_result.difference_update(term_docs)
            i += 2
        else:
            term_docs = set(find_term_postings(operator, inverted_index))
            current_result.intersection_update(term_docs)
            i += 1
            
    return current_result


def calculate_idf(inverted_index, total_documents):
    idf_scores = []
    for entry in inverted_index:
        term = entry[0]
        doc_count = len(entry) - 1
        idf_scores.append((term, np.log(total_documents / doc_count)))
    return idf_scores

def find_idf(term, idf_scores):
    for t, score in idf_scores:
        if t == term:
            return score
    return 0

"""def calculate_cosine_similarity(query_vector, doc_vector):
  
    dot_product = 0
    query_magnitude_sum_sq = 0
    doc_magnitude_sum_sq = 0
    
    # Create temporary sets for faster lookup
    query_terms = {t for t, _ in query_vector}
    doc_terms = {t for t, _ in doc_vector}

    common_terms = query_terms.intersection(doc_terms)
    
    # Simplified lookup by converting to dict for faster access
    query_dict = dict(query_vector)
    doc_dict = dict(doc_vector)
    
    for term in common_terms:
        dot_product += query_dict[term] * doc_dict[term]
        
    for _, weight in query_vector:
        query_magnitude_sum_sq += weight**2
    
    for _, weight in doc_vector:
        doc_magnitude_sum_sq += weight**2
        
    query_magnitude = np.sqrt(query_magnitude_sum_sq)
    doc_magnitude = np.sqrt(doc_magnitude_sum_sq)
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)
"""
def calculate_cosine_similarity(query_vector, doc_vector):
    
    all_terms = sorted(list(set([t for t, _ in query_vector] + [t for t, _ in doc_vector])))
    
    query_dict = dict(query_vector)
    doc_dict = dict(doc_vector)
    
    query_np_vector = np.array([query_dict.get(term, 0) for term in all_terms])
    doc_np_vector = np.array([doc_dict.get(term, 0) for term in all_terms])
    
    dot_product = np.dot(query_np_vector, doc_np_vector)
    
    query_magnitude = np.linalg.norm(query_np_vector)
    doc_magnitude = np.linalg.norm(doc_np_vector)
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)
def vector_space_retrieval(query, processed_documents, inverted_index, idf_scores):
    total_docs = len(processed_documents)
    query_tokens = preprocess_text(query)
    
    query_tf = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    query_vector = []
    for token, tf in query_tf.items():
        idf = find_idf(token, idf_scores)
        if idf > 0:
            query_vector.append((token, tf * idf))
            
    doc_scores = []
    
    for doc_id, doc_tokens in processed_documents:
        doc_tf = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
                
        doc_vector = []
        for token, tf in doc_tf.items():
            idf = find_idf(token, idf_scores)
            if idf > 0:
                doc_vector.append((token, tf * idf))
    
        score = calculate_cosine_similarity(query_vector, doc_vector)
        if score > 0:
            doc_scores.append((doc_id, score))
            
    ranked_docs = sorted(doc_scores, key=lambda item: item[1], reverse=True)
    return ranked_docs

def bim_retrieval(query, processed_documents, inverted_index):
    total_docs = len(processed_documents)
    query_terms = set(preprocess_text(query))
    
    doc_scores = []
    
    term_weights = {}
    for term in query_terms:
        nt = len(find_term_postings(term, inverted_index))
        if nt == 0:
            weight = np.log((total_docs - 0.5) / 0.5)
        else:
            weight = np.log((total_docs - nt + 0.5) / (nt + 0.5))
        term_weights[term] = weight

    for doc_id, doc_tokens in processed_documents:
        rsv = 0
        doc_terms = set(doc_tokens)
        
        for term in query_terms:
            if term in doc_terms:
                if term in term_weights:
                    rsv += term_weights[term]
        
        if rsv > 0:
            doc_scores.append((doc_id, rsv))
            
    ranked_docs = sorted(doc_scores, key=lambda item: item[1], reverse=True)
    return ranked_docs

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

def dice_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    size_sum = len(set1) + len(set2)
    if size_sum == 0:
        return 0
    return (2 * intersection) / size_sum

def run_all_models(data_directory, query):
    print("1. Loading and Preprocessing Documents...")
    documents_data = load_documents(data_directory)
    if not documents_data:
        print("Error: No .txt files found in the specified directory.")
        return

    processed_documents = []
    for doc_id, content in documents_data:
        processed_documents.append((doc_id, preprocess_text(content)))

    total_docs = len(processed_documents)
    all_doc_ids_set = {doc_id for doc_id, _ in processed_documents}
    print(f"Loaded and processed {total_docs} documents.")

    print("\n2. Building Inverted Index...")
    inverted_index = build_inverted_index(processed_documents)

    print(inverted_index)
    print("Inverted Index built.")

    print("\n--- Running Models for Query:", query, "---")
    
    # --- Extended Boolean Model ---
    print("\n[Extended Boolean Model]")
    boolean_results = boolean_retrieval_extended(query, inverted_index, all_doc_ids_set)
    if boolean_results:
        print(f"Documents found: {list(boolean_results)}")
    else:
        print("No documents found for the given query.")

    # --- Vector Space Model ---
    print("\n[Vector Space Model (VSM) with Cosine Similarity]")
    idf_scores = calculate_idf(inverted_index, total_docs)
    vsm_results = vector_space_retrieval(query, processed_documents, inverted_index, idf_scores)
    print("Ranked Documents (doc_id, score):")
    for doc_id, score in vsm_results:
        if isinstance(score, (int, float)):
            print(f"  - {doc_id}: {score:.4f}")
        else:
            print(f"  - {doc_id}: {score} (Error: Score is not a number)")

    # --- Binary Independence Model (BIM) ---
    print("\n[Binary Independence Model (BIM)]")
    bim_results = bim_retrieval(query, processed_documents, inverted_index)
    print("Ranked Documents (doc_id, score):")
    for doc_id, score in bim_results:
        if isinstance(score, (int, float)):
            print(f"  - {doc_id}: {score:.4f}")
        else:
            print(f"  - {doc_id}: {score} (Error: Score is not a number)")

    # --- Additional Similarity Measures (Example) ---
    print("\n[Example of Jaccard and Dice Similarity]")
    query_terms_set = set(preprocess_text(query))
    if vsm_results:
        top_doc_id = vsm_results[0][0]
        top_doc_terms_list = []
        for d_id, tokens in processed_documents:
            if d_id == top_doc_id:
                top_doc_terms_list = tokens
                break
        
        top_doc_terms_set = set(top_doc_terms_list)
        
        jaccard = jaccard_similarity(query_terms_set, top_doc_terms_set)
        dice = dice_similarity(query_terms_set, top_doc_terms_set)
        
        print(f"Similarity of query to top document '{top_doc_id}':")
        print(f"  - Jaccard Similarity: {jaccard:.4f}")
        print(f"  - Dice Similarity: {dice:.4f}")
    
# --- Execution ---
if __name__ == "__main__":
    if not os.path.exists('my_documents'):
        os.makedirs('my_documents')
        with open('my_documents/doc1.txt', 'w') as f:
            f.write("Information retrieval is an important field in computer science.")
        with open('my_documents/doc2.txt', 'w') as f:
            f.write("Machine learning and data science are related fields.")
        with open('my_documents/doc3.txt', 'w') as f:
            f.write("This document is about information retrieval and machine learning.")
        with open('my_documents/doc4.txt', 'w') as f:
            f.write("A basic text file for a document.")
    
    test_query = "information AND science OR learning NOT document"
    run_all_models('my_documents', test_query)import os
