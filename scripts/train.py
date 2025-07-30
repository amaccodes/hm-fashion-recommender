import torch
import torch.nn as nn

from src.model import LSTMModel
from src.utils import get_dataloaders

#config
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading data
train_loader, test_loader, num_articles, le = get_dataloaders(batch_size=BATCH_SIZE)

# modle, lossl optiizer
model = LSTMModel(num_articles, embedding_dim=50, hidden_dim=100, output_seq_len=7).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training model now
print(f"\nStarting training for {EPOCHS} epochs on device: {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        
        loss = criterion(outputs.view(-1, num_articles), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}")

# eval
print("\n Starting Evaluation ")
model.eval()
total_test_loss = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(x_batch)
        loss = criterion(outputs.view(-1, num_articles), y_batch.view(-1))
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Final Average Test Loss: {avg_test_loss:.4f}")

# save model
torch.save(model.state_dict(), 'lstm_model_final.pth')
print("\nModel training complete. Model saved to lstm_model_final.pth")
