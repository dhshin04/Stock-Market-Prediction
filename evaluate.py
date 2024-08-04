import torch, os, time
import numpy as np

from data_preprocessing import load_data
from model import StockPredictor
from config import *
import metrics
torch.manual_seed(1234)


def main():
    if no_test:
        print('Cannot evaluate since no test set available. Use predict.py instead.')
        
    else:
        if no_val:
            model_name = 'second_model.pth'
        else:
            model_name = 'first_model.pth'

        checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_name)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

        train_loader, validation_loader, test_loader = load_data(
            train_split, test_split, window_size, label_size, shift, train_batch, cv_batch, test_batch, no_val=no_val,
        )

        model = StockPredictor(
            out_channels, hidden_size, fc1_out, fc2_out, size, 
            dilation, num_layers, label_size, dropout_rate,
        )
        model.load_state_dict(checkpoint)
        model.to(DEVICE)

        if not no_val:
            # Validation Performance
            start = time.time()
            model.eval()            # Evaluation Mode: requires_grad=False, Batch Norm off
            accuracy_list = []      # List of accuracy (100 - MAPE)
            dir_acc_list = []

            with torch.no_grad():   # No gradient calculation
                for x_val, y_val in validation_loader:
                    x_val = x_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)

                    yhat = model(x_val)

                    if yhat.shape != y_val.shape:
                        raise Exception('yhat and y_test are not the same shape')

                    accuracy = metrics.accuracy(yhat, y_val, device=DEVICE)
                    dir_acc = metrics.directional_accuracy(x_val, yhat, y_val)
                    
                    accuracy_list.append(
                        np.mean(accuracy)       # Average accuracy of batch
                    )
                    dir_acc_list.append(
                        np.mean(dir_acc)
                    )
            
            end = time.time()
            print(f'Validation Accuracy: {np.mean(accuracy_list):.2f}%, Directional Accuracy: {np.mean(dir_acc_list):.2f}%, Elapsed Time: {end - start:.1f}s')

        # Test Performance
        start = time.time()
        accuracy_list = []      # List of accuracy (100 - MAPE)
        dir_acc_list = []

        with torch.no_grad():   # No gradient calculation
            for x_test, y_test in test_loader:
                x_test = x_test.to(DEVICE)
                y_test = y_test.to(DEVICE)

                yhat = model(x_test)

                if yhat.shape != y_test.shape:
                    raise Exception('yhat and y_test are not the same shape')

                accuracy = metrics.accuracy(yhat, y_test, device=DEVICE)
                dir_acc = metrics.directional_accuracy(x_test, yhat, y_test)
                
                accuracy_list.append(
                    np.mean(accuracy)       # Average accuracy of batch
                )
                dir_acc_list.append(
                    np.mean(dir_acc)
                )

        end = time.time()
        print(f'Test Accuracy: {np.mean(accuracy_list):.2f}%, Directional Accuracy: {np.mean(dir_acc_list):.2f}%, Elapsed Time: {end - start:.1f}s')


if __name__ == '__main__':
    main()
