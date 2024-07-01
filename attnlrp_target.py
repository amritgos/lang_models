import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Apply AttnLRP rules
attnlrp.register(model)

# Define the prompt
prompt = """\
Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# Generate the model's output
with torch.no_grad():
    output = model(input_ids=input_ids, use_cache=True)
generated_ids = output.logits.argmax(dim=-1)

# Decode the generated output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")

# Identify the target token from the generated text
target_token = "8,320"  # Replace this with the actual target token you're interested in
target_token_id = tokenizer.convert_tokens_to_ids(target_token)

# Find the token index of the target token in the generated text
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
target_indices = [i for i, token in enumerate(generated_tokens) if token == target_token]

# If there are multiple target tokens, you can choose one or aggregate across them
# Here we just take the first one for simplicity
target_token_index = target_indices[0]  # Get the index of the target token in the generated sequence

# Get the logits for the target token
target_logits = output.logits[0, target_token_index, :]

# Find the highest logit (target) to use for backward
max_target_logits, max_target_indices = torch.max(target_logits, dim=-1)

# Perform backward pass to compute gradients
model.zero_grad()
max_target_logits.backward(retain_graph=True)  # retain_graph=True to perform multiple backward passes if needed

# Compute relevance scores
relevance = input_embeds.grad.float().sum(-1).cpu()[0]

# Normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# Convert token IDs to token strings and clean them
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

# Map token relevance scores to words using whitespace tokenization
# Split on spaces and aggregate relevance scores
word_relevance = {}
current_word = []
current_word_scores = []

for token, score in zip(tokens, relevance):
    if token == ' ':
        if current_word:
            word = ''.join(current_word)
            word_relevance[word] = word_relevance.get(word, 0) + sum(current_word_scores)
            current_word = []
            current_word_scores = []
    else:
        current_word.append(token)
        current_word_scores.append(score.item())

# Add the last word if there are remaining tokens
if current_word:
    word = ''.join(current_word)
    word_relevance[word] = word_relevance.get(word, 0) + sum(current_word_scores)

# Sort words by relevance for visualization
sorted_words = sorted(word_relevance.items(), key=lambda x: x[1], reverse=True)
words, scores = zip(*sorted_words)

# Generate a heatmap for the word-level relevance scores
def plot_heatmap(scores, words, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    plt.figure(figsize=(12, 8))
    df = pd.DataFrame({'Word': words, 'Relevance': scores})
    df = df.sort_values(by='Relevance', ascending=False)
    sns.heatmap(df[['Word', 'Relevance']].set_index('Word').T, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Word-Level Relevance Heatmap')
    plt.savefig(filename)
    plt.show()

# Generate the heatmap
plot_heatmap(scores, words, "word_level_relevance_heatmap.pdf")

# Print target token for verification
print(f"Target Token: {tokenizer.convert_ids_to_tokens(target_token_id)}")
