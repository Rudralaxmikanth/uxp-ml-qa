import json
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, Trainer, TrainingArguments

# Load training data from JSON file
with open("training_data.json", "r") as file:
    train_dataset = json.load(file)

# Extract context, question, and answer from training data
train_contexts = [data['context'] for data in train_dataset]
train_questions = [data['question'] for data in train_dataset]
train_answers = [data['predicted_answer'] for data in train_dataset]

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize and format the training data
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)

# Train the model
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
)

trainer.train()

with open("documentation.doc", "r") as file:
    uxpdoc = file.read()

while True:
    my_input = input('Question:')
    if my_input == 'done':
        break

    encoded_input = tokenizer(uxpdoc, my_input, truncation=True, padding=True, return_tensors='pt')
    start_logits, end_logits = model(**encoded_input)

    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    answer = tokenizer.decode(encoded_input['input_ids'][0][start_index:end_index+1])

    print(f"answer: {answer}")
