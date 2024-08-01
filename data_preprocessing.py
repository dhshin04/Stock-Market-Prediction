''' Preprocess data to usable format by model '''
from torch.utils.data import DataLoader
from data.dataset import StockDataset
from data.read_csv import retrieve_sample, train_test_split


def create_datasets(train_split, test_split, window_size, label_size, shift):
    # Train-Val-Test Split
    stock_sample = retrieve_sample()
    csv_files = [stock_sample[0]]
    train_list, val_list, test_list = train_test_split(csv_files, train_split, test_split)

    training_set = StockDataset(train_list, window_size, label_size, shift)
    val_set = StockDataset(val_list, window_size, label_size, shift)
    test_set = StockDataset(test_list, window_size, label_size, shift)

    return training_set, val_set, test_set


def load_data(train_split, test_split, window_size, label_size, shift, train_batch, cv_batch, test_batch):
    '''
    Turn Dataset into DataLoader

    Arguments:
        train_batch (int): Batch size for train loader
        cv_batch (int): Batch size for validation loader
        test_batch (int): Batch size for test loader
    Returns:
        (tuple): DataLoaders for training, validation, and test sets
    '''

    # Load Dataset
    training_set, val_set, test_set = create_datasets(train_split, test_split, window_size, label_size, shift)

    workers = {
        'num_workers': 6,           # To speed up data transfer between CPU and GPU
        'persistent_workers': True, # Keep workers alive for next batch
        'pin_memory': True,         # Allocate tensors in page-locked memory for faster transfer
    }

    # Create Data Loaders
    train_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=training_set, 
        batch_size=train_batch,
        shuffle=True,
        **workers,
    )
    validation_loader = DataLoader(      
        dataset=val_set, 
        batch_size=cv_batch,
        shuffle=False,
        **workers,
    )
    test_loader = DataLoader(      
        dataset=test_set, 
        batch_size=test_batch,
        shuffle=False,
        **workers,
    )

    return train_loader, validation_loader, test_loader
