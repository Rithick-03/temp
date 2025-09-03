import pandas as pd
import nltk
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download("punkt_tab")
documents = [
    {"doc_id": "D1", "title": "Information requirement", "content": "query considers the user feedback as information requirement to search"},
    {"doc_id": "D2", "title": "Information retrieval", "content": "query depends on the model of information retrieval used"},
    {"doc_id": "D3", "title": "Prediction problem", "content": "Many problems in information retrieval can be viewed as prediction problems"},
    {"doc_id": "D4", "title": "Search", "content": "A search engine is one of applications of information retrieval models"},
    {"doc_id": "D5", "title": "Feedback", "content": "feedback is typically used by the system to modify the query and improve prediction"},
    {"doc_id": "D6", "title": "information retrieval", "content": "ranking in information retrieval algorithms depends on user query"}
]

ps = PorterStemmer()

def preprocess_text(text):
    tokens = [ps.stem(word.lower()) for word in word_tokenize(text) if word.isalnum()]
    return tokens

def build_inverted_index(docs, stop_words=None):
    inverted_index = defaultdict(list)
    for doc in docs:
        tokens = preprocess_text(doc["content"])
        if stop_words:
            tokens = [token for token in tokens if token not in stop_words]
        for token in set(tokens):
            inverted_index[token].append(doc["doc_id"])
    return inverted_index

def compute_stop_words(docs):
    all_tokens = []
    for doc in docs:
        all_tokens.extend(preprocess_text(doc["content"]))
    term_freq = Counter(all_tokens)
    stop_words = {term for term, _ in term_freq.most_common(10)}
    return stop_words

def compute_index_size(inverted_index):
    num_terms = len(inverted_index)
    num_postings = sum(len(postings) for postings in inverted_index.values())
    return num_terms, num_postings

def evaluate_boolean_query(query, inverted_index, all_docs):
    tokens = query.lower().split()
    if not tokens:
        return set()

    all_doc_ids = set(doc["doc_id"] for doc in all_docs)
    result = None
    operator = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ('and', 'or', 'not'):
            operator = token
            i += 1
            continue

        term = ps.stem(token)
        current_docs = set(inverted_index.get(term, []))

        if operator == 'not':
            current_docs = all_doc_ids - current_docs
            operator = None

        if result is None:
            result = current_docs
        elif operator == 'and':
            result = result.intersection(current_docs)
            operator = None
        elif operator == 'or':
            result = result.union(current_docs)
            operator = None

        i += 1

    return result if result is not None else set()

def main():
    initial_index = build_inverted_index(documents)
    initial_terms, initial_postings = compute_index_size(initial_index)

    stop_words = compute_stop_words(documents)
    filtered_index = build_inverted_index(documents, stop_words)
    filtered_terms, filtered_postings = compute_index_size(filtered_index)

    print(f"Stop Words: {stop_words}")
    print(f"\nInitial Index Size: {initial_terms} terms, {initial_postings} postings")
    print(f"Filtered Index Size (without stop words): {filtered_terms} terms, {filtered_postings} postings")

    index_data = [{"Term": term, "Postings": ", ".join(postings)} for term, postings in filtered_index.items()]
    df_index = pd.DataFrame(index_data)
    print("\nInverted Index (without stop words):")
    print(df_index.to_string(index=False))

    sample_queries = [
        "query AND feedback",
        "retrieval OR prediction",
        "search AND NOT engine",
        "information AND retrieval NOT prediction"
    ]

    print("\nBoolean Query Results:")
    query_results = []
    for query in sample_queries:
        result = evaluate_boolean_query(query, filtered_index, documents)
        query_results.append({"Query": query, "Matching Documents": ", ".join(sorted(result)) if result else "None"})

    df_queries = pd.DataFrame(query_results)
    print(df_queries.to_string(index=False))

if __name__ == "__main__":
    main()
