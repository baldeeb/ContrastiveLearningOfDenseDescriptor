import torch
import os

def save_model(path, model, optimizer=None):
    '''
    Saves model and optimizer.
    NOTE: optimizer is currently assumed to be Adam.
    '''
    checkpoint = {'model_state_dict': model.state_dict()}
    checkpoint['model_config'] = model.congifuration()
    if optimizer: checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, os.path.abspath(path))

from models.dense_model import DenseModel
def load_dense_model(path, device='cuda'):
    model_dict = torch.load(path)
    model = DenseModel(model_dict['model_config'], False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.load_state_dict(model_dict['model_state_dict'])
    optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model, optimizer