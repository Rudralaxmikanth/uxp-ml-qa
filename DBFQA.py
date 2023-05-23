from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

# Set the model to evaluation mode
model.eval()

# Define your context and question
context = "Hello Everyone, my name is Rudra Laxmi Kanth, and I am a 3rd year undergraduate from IIT Madras. I joined Adobe on May 15, 23."
question = "When did you join Adobe?"

# Tokenize the context and question
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

# Retrieve the input IDs and attention mask
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Perform the question answering
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

# Find the start and end positions with the highest scores
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

# Convert the token indices back to text
all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])

print("Question:", question)
print("Answer:", answer)
