import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import clean_tokens
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceLLM
from langchain.prompts import PromptTemplate

def plot_relevance_heatmap(tokens, relevance):
    plt.figure(figsize=(12, 2))
    ax = sns.heatmap(relevance.unsqueeze(0), annot=[tokens], fmt='', cmap='coolwarm', cbar=False)
    ax.set_xticklabels(tokens, rotation=90)
    plt.show()

def compute_and_plot_word_level_relevance(prompt):
    # Load the model and tokenizer using LangChain
    llm = HuggingFaceLLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create the LangChain
    template = PromptTemplate(template=prompt)
    chain = LLMChain(llm=llm, prompt=template)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(llm.device)
    input_embeds = llm.model.get_input_embeddings()(input_ids)

    # Apply AttnLRP rules
    attnlrp.register(llm.model)

    # Generate the full output logits
    output_logits = llm.model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

    # Tokenize the output text
    output_ids = torch.argmax(output_logits, dim=-1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Compute gradients and relevance scores
    relevance_scores = []
    for i in range(output_logits.size(1)):
        max_logits = output_logits[0, i, :].max()
        max_logits.backward(retain_graph=True)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0]
        relevance_scores.append(relevance)
        llm.model.zero_grad()

    # Average relevance across all tokens
    average_relevance = torch.mean(torch.stack(relevance_scores), dim=0)

    # Normalize relevance between [-1, 1] for plotting
    average_relevance = average_relevance / average_relevance.abs().max()

    # Remove '_' characters from token strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    # Aggregate relevance at the word level
    word_relevances = []
    current_word_relevance = 0
    current_word_tokens = []
    words = []

    for token, rel in zip(tokens, average_relevance):
        if not token.startswith("##"):
            if current_word_tokens:
                # Append average relevance for the current word
                word_relevances.append(current_word_relevance / len(current_word_tokens))
                words.append("".join(current_word_tokens))
            current_word_relevance = 0
            current_word_tokens = []
        current_word_relevance += rel
        current_word_tokens.append(token)

    if current_word_tokens:
        # Append the last word's average relevance
        word_relevances.append(current_word_relevance / len(current_word_tokens))
        words.append("".join(current_word_tokens))

    # Convert relevance list to a tensor for plotting
    word_relevances = torch.tensor(word_relevances)

    # Plot the relevance heatmap
    plot_relevance_heatmap(words, word_relevances)

    return output_text, word_relevances
