import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo GPT-2 y el tokenizador
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_gpt2_response(prompt, max_length=50, num_return_sequences=1):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, do_sample=True, temperature=0.7)
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses
