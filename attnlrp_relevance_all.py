import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Apply AttnLRP rules
attnlrp.register(model)

prompt = """\
Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

relevances = []

for token_idx in range(output_logits.size(1)):  # Loop over each token position
    max_logits, _ = torch.max(output_logits[0, token_idx, :], dim=-1)
    max_logits.backward(retain_graph=True)  # Retain computation graph for multiple backward passes
    relevance = input_embeds.grad.float().sum(-1).cpu()[0]
    relevance = relevance / relevance.abs().max()  # Normalize relevance
    relevances.append(relevance)
    input_embeds.grad.zero_()  # Clear gradients for next iteration

# Clean tokens for plotting
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

# Plot relevance for the last token
plot_relevance_heatmap(tokens, relevances[-1])

# Decode the generated text
output_text = tokenizer.decode(output_logits.argmax(dim=-1)[0].tolist(), skip_special_tokens=True)
print("Generated Text:", output_text)
