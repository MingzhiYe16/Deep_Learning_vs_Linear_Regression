import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Load and prepare data
df = pd.read_csv('simtrain5.txt', sep='\s+')
# Convert numerical data to text format for T5 processing
df['input_text'] = df.apply(
    lambda row: ' '.join(['x' + str(i + 1) + ' ' + str(row['x' + str(i + 1)]) for i in range(13)]), axis=1)
df['target_text'] = df['y'].apply(str)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2)


class NumDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_token_len=128):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        data_row = self.dataframe.iloc[index]

        input_text = data_row.input_text
        target_text = data_row.target_text

        input_encoding = tokenizer(
            input_text,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = tokenizer(
            target_text,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        return dict(
            input_ids=input_encoding['input_ids'].flatten(),
            attention_mask=input_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )


tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

train_dataset = NumDataset(train_df, tokenizer)
val_dataset = NumDataset(val_df, tokenizer)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss

    avg_val_loss = total_loss / len(val_dataloader)
    print(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}')

    # Save the best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'Best_T5_model.pth')
        print("Saved Best Model")

print("Training complete.")
