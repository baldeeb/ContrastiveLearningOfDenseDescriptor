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
