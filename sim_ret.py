from collections import Counter
import re

def compute_highlight_scores(document, query, window_size=50):
    """
    Compute contextualized highlight scores for tokens in a document.
    
    Parameters:
    - document (str): The full text of the document.
    - query (str): The search query string.
    - window_size (int): The size of the context window around query terms to consider.
    
    Returns:
    - highlights (list of tuples): Each tuple contains (start_index, end_index, score) for a highlight.
    """
    def tokenize(text):
        """Tokenize the text into words."""
        return re.findall(r'\b\w+\b', text.lower())

    def score_token(token, query_tokens):
        """Compute a simple score for a token based on its presence in the query."""
        return 1 if token in query_tokens else 0

    # Tokenize document and query
    doc_tokens = tokenize(document)
    query_tokens = set(tokenize(query))
    
    # Compute term frequency for each token in the document
    term_freq = Counter(doc_tokens)
    
    # Compute highlight scores
    highlights = []
    for i, token in enumerate(doc_tokens):
        if score_token(token, query_tokens) > 0:
            start_index = max(0, i - window_size // 2)
            end_index = min(len(doc_tokens), i + window_size // 2 + 1)
            snippet = ' '.join(doc_tokens[start_index:end_index])
            score = score_token(token, query_tokens) * term_freq[token]
            highlights.append((start_index, end_index, score, snippet))
    
    # Sort highlights by score (higher scores first)
    highlights.sort(key=lambda x: x[2], reverse=True)
    
    return highlights

# Example usage
document = "Python is an interpreted, high-level programming language. Python is popular for web development and data science."
query = "Python data science"
highlights = compute_highlight_scores(document, query)

for start_idx, end_idx, score, snippet in highlights:
    print(f"Highlight Score: {score}")
    print(f"Context: {snippet}")
    print()
