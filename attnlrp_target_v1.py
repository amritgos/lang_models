import torch
from lxt.models import LlamaForQuestionAnswering
from lxt.tokenizer import LlamaTokenizer

# Initialize the Llama model and tokenizer
model_name = "lxt-llama-base"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForQuestionAnswering.from_pretrained(model_name)

# Define a function for attnLRP
def attn_lrp(model, input_ids, attention_mask, target_word_ids):
    model.eval()

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions

    # Initialize relevance for the target words
    relevance = torch.zeros_like(input_ids, dtype=torch.float)
    for idx in target_word_ids:
        relevance[:, idx] = 1

    # Backpropagate the relevance through the layers
    for layer in reversed(range(len(attentions))):
        attention = attentions[layer]
        relevance = torch.matmul(attention.transpose(-1, -2), relevance.unsqueeze(-1)).squeeze(-1)

    return relevance

# Example usage
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris."

inputs = tokenizer(question, context, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Assume the answer is "Paris"
answer = "Paris"
answer_ids = tokenizer.encode(answer, add_special_tokens=False)

# Identify target word indices in the generated answer
generated_answer = model.generate(input_ids=input_ids, attention_mask=attention_mask)
decoded_answer = tokenizer.decode(generated_answer[0], skip_special_tokens=True)
target_word_ids = [i for i, token in enumerate(generated_answer[0]) if token in answer_ids]

# Get relevance scores
relevance_scores = attn_lrp(model, input_ids, attention_mask, target_word_ids)

# Print relevance scores
print("Relevance scores:", relevance_scores)
