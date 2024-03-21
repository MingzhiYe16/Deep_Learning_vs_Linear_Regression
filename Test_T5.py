import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader

## Load the Test Dataset
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

# Load test data
test_df = pd.read_csv('simtest5.txt', sep='\s+')
test_df['input_text'] = test_df.apply(lambda row: ' '.join(['x'+str(i+1)+' '+str(row['x'+str(i+1)]) for i in range(13)]), axis=1)
test_df['target_text'] = test_df['y'].apply(str)  # Might not be needed for predictions, but useful if available

test_dataset = NumDataset(test_df, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32)  # Adjust batch_size as needed

## Load the Saved Model

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load('//Best_T5_model.pth'))
model.eval()  # Set to eval mode for predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



## Generate Predictions

model.eval()  # Ensure model is in eval mode
predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend([float(pred) for pred in decoded_preds])

## Correct Actuals Extraction
# Directly use y values from the test DataFrame
actuals = test_df['y'].tolist()




## Calculate MSE and R²
mse = mean_squared_error(actuals, predictions)
r_square = r2_score(actuals, predictions)

print(f'MSE: {mse}')
print(f'R-squared: {r_square}')

# MSE: 0.0015321381944504474
# R-squared: 0.09062966991117005