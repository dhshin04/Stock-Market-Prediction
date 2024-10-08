# TODO: Continue testing different hyperparameters to reduce MAPE

import torch
torch.manual_seed(1234)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Hyperparameters ##
# Data Loaders
start_date = '2000-01-03'       # Train set must start here or before
train_split = '2010-01-04'      # Train set ends here, val set starts here
test_split = '2015-01-02'       # Val set ends here, test set starts here
window_size = 480               # Approximately 12 months of trading time
label_size = 20                 # Approximately 1 month of trading time
shift = 1                       # Sequence shift by 1 day
train_batch = 32                # Train Loader Batch Size
cv_batch = 256                  # Validation Loader Batch Size
test_batch = 64                 # Test Loader Batch Size

# Model Definition
out_channels = 96               # Units in CNN
hidden_size = 125               # Units in LSTM
fc1_out = 64                    # Units in first FC layer
fc2_out = 32                    # Units in second FC layer
size = 3                        # Kernel size for CNN
dilation = 1                    # Dilation for CNN
num_layers = 1                  # Number of LSTM layers
dropout_rate = 0.25              # For Dropout Regularization

# Train Loop
learning_rate = 2e-6            # For Training - best 3e-6, try 1e-6 and 1e-5
epochs = 60                     # For Training
warmup_step = 10                # For LambdaLR Warmup
weight_decay = 0.0005           # For AdamW 
T_max = epochs                  # For CosineAnnealingLR
eta_min = learning_rate / 30    # For CosineAnnealingLR

# Train Variation
no_val = True                   # Use validation set for training as well
no_test = False                 # Use test set for training as well
