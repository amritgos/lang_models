from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the tokenizer (example with BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example text
text = "The Apollo program was a series of manned spaceflight missions carried out by NASA."

# Tokenize the text with special tokens
tokenized_input = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'].squeeze())
offsets = tokenized_input['offset_mapping'].squeeze().tolist()

# Create a mapping from tokens to words
words = text.split()
word_offsets = []
for word in words:
    start = text.index(word)
    end = start + len(word)
    word_offsets.append((start, end))

token_to_word_mapping = []
for token_start, token_end in offsets:
    if token_start == token_end:  # Special tokens like [CLS], [SEP]
        token_to_word_mapping.append("[SPECIAL]")
    else:
        for i, (word_start, word_end) in enumerate(word_offsets):
            if token_start >= word_start and token_end <= word_end:
                token_to_word_mapping.append(words[i])
                break

# Example relevance tensor (for demonstration purposes)
relevance_tensor = np.random.rand(len(tokens))

# Aggregate relevance scores for each word
word_relevance = {}
for token, relevance, word in zip(tokens, relevance_tensor, token_to_word_mapping):
    if word not in word_relevance:
        word_relevance[word] = 0
    word_relevance[word] += relevance

# Normalize relevance scores
words = list(word_relevance.keys())
relevance_scores = np.array(list(word_relevance.values()))
normalized_relevance = (relevance_scores - np.min(relevance_scores)) / (np.max(relevance_scores) - np.min(relevance_scores))

# Select top 20 words based on relevance scores
top_indices = np.argsort(normalized_relevance)[-20:]
top_words = [words[i] for i in top_indices]
top_relevance = normalized_relevance[top_indices]

# Reshape relevance_tensor if needed (e.g., if it's 1D)
top_relevance = top_relevance.reshape(1, -1)

# Plot heatmap
plt.figure(figsize=(14, 4))
heatmap = sns.heatmap(top_relevance, cmap="YlGnBu", xticklabels=top_words, cbar=True, annot=True)
heatmap.set_title("Top 20 Word Attributions")
heatmap.set_xlabel("Words")
heatmap.set_ylabel("Relevance Score")
heatmap.set_xticklabels(top_words, rotation=45, ha="right")  # Rotate tokens for better readability

# Add a color bar legend
cbar = heatmap.collections[0].colorbar
cbar.set_label('Relevance Score')

plt.tight_layout()
plt.show()
