import torch
from model import LSTMModel
from utils import get_dataloaders
import numpy as np
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data + label encoder
_, test_loader, num_articles, le = get_dataloaders(batch_size=1)

model = LSTMModel(num_articles, embedding_dim=50, hidden_dim=100, output_seq_len=7).to(DEVICE)
MODEL_PATH = os.path.join("data", "processed", "final_lstm_data.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_sample = x_batch.to(DEVICE)
        y_true_indices = y_batch.cpu().numpy()[0]
        output = model(x_sample)
        predicted_indices = torch.argmax(output, dim=-1).cpu().numpy()[0]

        # Clip predicted indices to valid range
        max_index = len(le.classes_) - 1
        predicted_indices = np.clip(predicted_indices, 0, max_index)
        y_true_indices = np.clip(y_true_indices, 0, max_index)

        # Decode to article IDs
        y_true_articles = le.inverse_transform(y_true_indices)
        predicted_articles = le.inverse_transform(predicted_indices)

        print("Predicted article IDs:", predicted_articles)
        print("True article IDs:", y_true_articles)
        break
