"""
From GAT utils
"""
import torch as th
import os

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        th.save(model.state_dict(), os.path.join(self.checkpoint_path, "best.pt"))

    def load_checkpiont(self):
        return th.load(os.path.join(self.checkpoint_path, "best.pt"))