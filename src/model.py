from torch.utils.data import DataLoader
from transformers import AdamW, BertForQuestionAnswering, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data_loader import data_loader.LegalDataset
import torch

def main():
    # Load the JSON data
    file_path = './data/your_file.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Create dataset
    train_dataset = LegalDataset(data, tokenizer)

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    # DataLoader with collate_fn for dynamic padding
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

    # Load the model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Training loop
    model.train()
    epochs = 3

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./Fine_tuneBert')
    tokenizer.save_pretrained('./Fine_tuneBert')
    print("Model and tokenizer have been saved to './Fine_tuneBert'.")

if __name__ == "__main__":
    main()