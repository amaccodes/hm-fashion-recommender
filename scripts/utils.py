import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def get_dataloaders(batch_size=64):
    """
    Loads data and returns training/testing dataloaders
    """ 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data/processed/final_lstm_data.pth')
    print(f"base_dir: {base_dir}")
    print(f"data_path: {data_path}")
    print(f"File exists? {os.path.exists(data_path)}")

    data_payload = torch.load(data_path)
    
    X_train = data_payload['X_train']
    Y_train = data_payload['Y_train']
    X_test = data_payload['X_test']
    Y_test = data_payload['Y_test']
   
    article_to_idx = data_payload['article_to_idx']
    idx_to_article = data_payload['idx_to_article']

    le = LabelEncoder()
    le.classes_ = np.array([idx_to_article[i] for i in range(len(idx_to_article))])

    num_articles = len(le.classes_)
    print(f"Data loaded. Found {num_articles} unique articles.")

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_articles, le
