import torch
import torch.nn as nn

from model import LSTMModel
from utils import get_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load split data from utils.py
_, test_loader, num_articles, _ = get_dataloaders(batch_size=64)

# Load model weights (make sure this file exists, run train.py if not)
model = LSTMModel(num_articles, embedding_dim=50, hidden_dim=100, output_seq_len=7).to(DEVICE)
model.load_state_dict(torch.load('final_lstm_data.pth', map_location=DEVICE))

# Display one prediction
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_sample = x_batch[0].unsqueeze(0).to(DEVICE)  # Take the first sample
        y_true = y_batch[0].cpu().numpy()
        output = model(x_sample)
        predicted = torch.argmax(output, dim=-1).cpu().numpy()
        print("Sample prediction:", predicted)
        print("True label:", y_true)
        break  # Only show one prediction
