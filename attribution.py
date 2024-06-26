import torch
from transformers import LLAMAForQuestionAnswering, LLAMATokenizer
import captum.attr as attr

# Load LLAMA2 model and tokenizer
model_name = "allenai/llama2-large-uncased"
tokenizer = LLAMATokenizer.from_pretrained(model_name)
model = LLAMAForQuestionAnswering.from_pretrained(model_name)
model.eval()

# Example context and question
context = "Context: Add your context here."
question = "Question: Add your question here?"

# Tokenize inputs
inputs = tokenizer(context, question, return_tensors="pt")

# Define attribution method using Layer Integrated Gradients (IG)
attribution_method = attr.LayerIntegratedGradients(model, model.base_model.embeddings)

# Compute attributions
attributions_start, delta_start = attribution_method.attribute(inputs["input_ids"], 
                                                              additional_forward_args=(inputs["token_type_ids"], inputs["attention_mask"]),
                                                              target=inputs["input_ids"].unsqueeze(0))

# Interpret attributions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

# Print or visualize attributions
print("Top tokens contributing to the start position:")
for token, attr_score in sorted(zip(tokens, attributions_start[0].tolist()), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{token}: {attr_score}")

# Example: Get predicted answer span
with torch.no_grad():
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1
answer_tokens = inputs["input_ids"][0][start_index:end_index]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"\nQuestion: {question}\nPredicted Answer: {answer}")
