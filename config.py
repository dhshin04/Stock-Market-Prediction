import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Hyperparameters ##
# Data Loaders
train_split = '2012-01-03'      # Train set ends here, val set starts here
test_split = '2016-01-04'       # Val set ends here, test set starts here
window_size = 140               # Approximately 7 months of trading time
label_size = 20                 # Approximately 1 month of trading time
shift = 1                       # Sequence shift by 1 day
train_batch = 48                # Train Loader Batch Size
cv_batch = 48                   # Validation Loader Batch Size

# Model Definition
out_channels = 64               # Units in CNN
hidden_size = 64                # Units in LSTM
fc1_out = 256                   # Units in first FC layer
fc2_out = 128                   # Units in second FC layer
size = 3                        # Kernel size for CNN
dilation = 1                    # Dilation for CNN
num_layers = 4                  # Number of LSTM layers
dropout_rate = 0.25             # For Dropout Regularization

# Train Loop
accumulation_size = 1           # For Gradient Accumulation
learning_rate = 1e-6            # For Training
epochs = 100                    # For Training
warmup_step = 10                # For LambdaLR Warmup
weight_decay = 0.0005           # For AdamW 
T_max = epochs                  # For CosineAnnealingLR
eta_min = learning_rate / 30    # For CosineAnnealingLR