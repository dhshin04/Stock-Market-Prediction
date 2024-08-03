from config import *        # Contains DEVICE and hyperparameters
from data_preprocessing import load_data
from model import StockPredictor
import metrics

import os, time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.multiprocessing as mp
from torch.amp import GradScaler
torch.manual_seed(1234)

# Enable TF32 mode for matrix multiplications - ensure speed + accuracy for Mixed Precision
torch.backends.cuda.matmul.allow_tf32 = True


def warmup(epoch):
    '''
    For LambdaLR. Warms up learning rate for smoother convergence.

    Arguments:
        epoch (int): Current epoch 
    Returns:
        (float): Scaled learning rate
    '''
    if epoch < warmup_step:
        return float(epoch) / warmup_step
    return 1.


def train_one_epoch(epoch, model, train_loader, validation_loader, scaler, criterion, optimizer, scheduler):
    # Train
    model.train()               # Train Mode: requires_grad=True, Batch Norm on
    train_loss_list = []        # Contains train loss per batch

    start = time.time()
    for x, y in train_loader:   # Mini-Batch Gradient Descent
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if scaler is not None and torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):   # Mixed Precision
                yhat = model(x)        # Returns dictionary of losses
                loss = torch.sqrt(criterion(yhat, y))
        else:
            yhat = model(x)
            loss = torch.sqrt(criterion(yhat, y))
        
        train_loss_list.append(loss.item())
        
        if scaler is not None and torch.cuda.is_available():
            scaler.scale(loss).backward()         # Compute gradient of loss
            scaler.step(optimizer)     # Update parameters
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()      # clear old gradient before new gradient calculation
        
    # Evaluate
    model.eval()            # Evaluation Mode: requires_grad=False, Batch Norm off
    cv_loss_list = []       # Contains val loss per batch
    accuracy_list = []

    with torch.no_grad():   # No gradient calculation
        for x_test, y_test in validation_loader:
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)

            yhat = model(x_test)
            loss = torch.sqrt(criterion(yhat, y_test))      # RMSE Loss

            if yhat.shape != y_test.shape:
                raise Exception('yhat and y_test are not the same shape')

            accuracy = metrics.accuracy(yhat, y_test, device=DEVICE)
            
            cv_loss_list.append(loss.item())
            accuracy_list.append(
                np.mean(accuracy)       # Average accuracy of batch
            )
    
    end = time.time()
    print(f'Epoch: {(epoch + 1)}/{epochs}, Training Loss: {np.mean(train_loss_list):.4f}, Validation Loss: {np.mean(cv_loss_list):.4f}, Average Accuracy: {np.mean(accuracy_list):.2f}%, Elapsed Time: {end - start:.1f}s')
    
    scheduler.step()            # Next step for warmup


def main():
    ''' Data Loading '''
    train_loader, validation_loader, _ = load_data(
        start_date, train_split, test_split, window_size, label_size, shift, train_batch, cv_batch,
    )

    model = StockPredictor(
        out_channels, hidden_size, fc1_out, fc2_out, size, dilation, num_layers, label_size, dropout_rate,
    )

    model.to(DEVICE)

    ''' Model Training and Evaluation '''
    # Compile Model
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)                     # LR Warmup to Complement AdamW
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)   # LR Scheduler to Complement AdamW
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[warmup_step])
    
    if torch.cuda.is_available():
        scaler = GradScaler()   # Mixed Precision for faster training
    else:
        scaler = None
    
    # Train and Evaluate Model
    print('Training...')
    for epoch in range(epochs):
        train_one_epoch(
            epoch, model, train_loader, validation_loader, 
            scaler, criterion, optimizer, scheduler,
        )

    # Save Model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'model.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)        # for compatibility with Windows multiprocessing
    main()
