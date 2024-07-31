from torch.utils.data import Dataset


class StockDataset(Dataset):
    
    def __init__(self, series_list, window_size, label_size, shift):
        '''
        Arguments:
            series_list (list): List of series tensors for each stock
            window_size (int): Total window size for each sequence
            label_size (int): Size of just the labels part of sequence
            shift (int): Gap between each sequence (in days)
        '''
        self.series_list = series_list
        self.window_size = window_size
        self.label_size = label_size
        self.shift = shift
        self.len_list = []       # List of each series' length
        for series in self.series_list:
            self.len_list.append(
                (len(series) - self.window_size) // self.shift + 1
            )
    
    def __getitem__(self, idx):
        # Find appropriate series
        for i, length in enumerate(self.len_list):
            if idx < length:
                series = self.series_list[i]
                break
            idx -= length
        
        # Find appropriate sequence
        start = idx * self.shift        # Starting position (index) of sequence
        sequence = series[start:(start + self.window_size)]

        # Divide sequence into features and labels
        features = sequence[:-self.label_size]
        labels = sequence[-self.label_size:]

        return features, labels

    def __len__(self):
        return sum(self.len_list)
