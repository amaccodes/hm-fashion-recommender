
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_articles, embedding_dim=50, hidden_dim=100, output_seq_len=7):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_articles, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_articles * output_seq_len)
        self.output_seq_len = output_seq_len
        self.num_articles = num_articles

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        out = out.view(-1, self.output_seq_len, self.num_articles)
        return out
