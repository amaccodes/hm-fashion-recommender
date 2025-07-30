

import torch
from torch.utils.data import TensorDataset, DataLoader

def get_dataloaders(batch_size=64):
    """
    Loads data and returns training/testing dataloaders
    """ 
    data_payload = torch.load('data/processed/final_lstm_data.pth')
    
    X_train = data_payload['X_train']
    Y_train = data_payload['Y_train']
    X_test = data_payload['X_test']
    Y_test = data_payload['Y_test']
    le = data_payload['label_encoder']
    
    num_articles = len(le.classes_)
    print(f"Data loaded. Found {num_articles} unique articles.")

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_articles, le
