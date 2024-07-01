import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp

# Initialize the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
outputs = model(input_ids=input_ids, use_cache=True, output_attentions=True)

# Get the generated answer tokens
generated_ids = outputs.logits.argmax(dim=-1)
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
generated_answer = tokenizer.decode(generated_ids[0])

# Extract the attention weights
attentions = outputs.attentions

# Compute relevance scores from the attention weights
attention = torch.stack(attentions).mean(dim=1)  # Average over all attention heads
attention_scores = attention[:, 0, 1:]  # Get attention weights from prompt tokens to the output tokens

# Get the relevance scores for the context tokens
context_attention_scores = attention_scores.mean(dim=-1).cpu().detach().numpy()

# Convert token IDs to token strings
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Create a dictionary to map tokens to their relevance scores
token_relevance = {token: score for token, score in zip(tokens, context_attention_scores)}

# Sort tokens by relevance for visualization
sorted_tokens = sorted(token_relevance.items(), key=lambda x: x[1], reverse=True)
tokens, scores = zip(*sorted_tokens)

# Define a function to plot the attention heatmap
def plot_attention_heatmap(tokens, scores, filename):
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame({'Token': tokens, 'Relevance': scores})
    df = df.sort_values(by='Relevance', ascending=False)
    sns.heatmap(df[['Token', 'Relevance']].set_index('Token').T, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Token-Level Relevance Heatmap')
    plt.savefig(filename)
    plt.show()

# Generate the heatmap
plot_attention_heatmap(tokens, scores, "token_level_relevance_heatmap.pdf")

# Print a few of the most relevant tokens for verification
print("Generated Answer:")
print(generated_answer)

print("\nMost relevant tokens and their scores:")
for token, score in sorted_tokens[:10]:
    print(f"{token}: {score:.4f}")
