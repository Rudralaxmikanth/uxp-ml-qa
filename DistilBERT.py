from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import torch
from torch.utils.data import Dataset, DataLoader
import json

with open("documentation.doc", "r") as file:
    uxpdoc = file.read()

# Load the dataset
with open("qa_dataset.json", "r") as file:
    dataset = json.load(file)

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        context = example["context"]
        qas = example["qas"]
        
        inputs = []
        start_positions = []
        end_positions = []
        
        for qa in qas:
            question = qa["question"]
            id = qa["id"]
            
            # Tokenize the context and question
            encoding = self.tokenizer(question, context, truncation=True, padding="max_length", max_length=256)
            
            inputs.append(encoding.input_ids)
            
            # Find the start and end positions of the answer in the tokenized inputs
            answer_start = context.find(qa["answer"])
            answer_end = answer_start + len(qa["answer"]) - 1
            
            # Check if the answer is present in the context
            if answer_start != -1:
                # Convert character positions to token positions
                start_position = encoding.char_to_token(answer_start)
                end_position = encoding.char_to_token(answer_end)
                
                start_positions.append(start_position)
                end_positions.append(end_position)
        
        if not start_positions:
            # If no valid start positions found, return None for all fields
            return None
        
        return {
            "input_ids": torch.tensor(inputs),
            "start_positions": torch.tensor(start_positions),
            "end_positions": torch.tensor(end_positions)
        }




# Create an instance of the dataset
qa_dataset = QADataset(dataset)

# Create a DataLoader for batching and shuffling the data during training
dataloader = DataLoader(qa_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)

# Fine-tune a pre-trained model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {average_loss:.4f}")

# Create a question-answering pipeline with the fine-tuned model
question_answers = QuestionAnsweringPipeline(model, tokenizer="distilbert-base-uncased")

# Test the pipeline
while True:
    user_input = input("Question: ")
    if user_input == "done":
        break
    
    preds = question_answers(user_input, context=uxpdoc)
    
    print(f"Answer: {preds['answer']}")
    print(f"Score: {round(preds['score'], 4)}")
    print(f"Start: {preds['start']}")
    print(f"End: {preds['end']}")
