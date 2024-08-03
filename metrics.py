import torch
import numpy as np


def accuracy(yhat, y, device):
    '''
    Calculates 100 - MAPE (mean average percent error) %

    Arguments:
        yhat (torch.tensor): Shape = (batch_size, label_size)
        y (torch.tensor): Same shape as yhat, represents target label
        device (str): GPU (or CPU)
    Returns:
        (np.ndarray): NumPy array of each accuracy in batch (in percentage)
    '''

    accuracy = torch.max(
        100 - torch.abs((yhat - y) / (y + 1e-8)) * 100,   # 1e-8 to prevent division by zero
        torch.tensor(0.0, device=device)    # Lower bound
    )

    return accuracy.to('cpu').numpy()


def directional_accuracy(x, yhat, y):
    '''
    Calculates directional accuracy (percentage of correct direction predicted)

    Arguments:
        x (torch.tensor): Input; shape = (batch_size, window_size - label_size, num_features)
        yhat (torch.tensor): Prediction; shape = (batch_size, label_size)
        y (torch.tensor): Target label; shape = (batch_size, label_size)
    Returns:
        (np.ndarray): NumPy array of each directional accuracy in batch (in percentage)
    '''

    previous = x[:, -1, 3]     # Closing price of previous day for each batch
    accuracy_list = []
    for prev, pred, target in zip(previous, yhat, y):       # For each batch
        correct = 0     # Number of correct directions
        if len(pred) != len(target):
            raise ValueError('Different lengths of yhat and y')
        total = len(pred)

        if (prev < pred[0] and prev < target[0]) or (prev >= pred[0] and prev >= target[0]):
            correct += 1

        for i in range(1, len(pred)):      # For each prediction in batch
            if (pred[i - 1] < pred[i] and target[i - 1] < target[i]) or (pred[i - 1] >= pred[i] and target[i - 1] >= target[i]):
                correct += 1
        accuracy_list.append((correct / total * 100))

    return np.array(accuracy_list)
