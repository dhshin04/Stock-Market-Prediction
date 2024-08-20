# Stock-Market-Prediction

**Forecasting Stock Market Prices Using Historical Data**

This repository contains code for a prediction algorithm designed to forecast the next 20 days of closing prices for tech stocks using historical data (e.g., closing prices, volume).

## Project Description

### Project Status: _Ongoing_

_Frameworks and Tools: PyTorch, AWS, Nvidia NGC, Gemini API_

_\*\*AWS EC2 and Nvidia NGC used to leverage CUDA for faster training_

This project aims to identify general patterns in the stock market that can be leveraged to predict future stock performance. To achieve this, 15 tech stocks are selected from diverse industry sectors using the Gemini API. The Gemini API provides a dynamic selection of tech stocks, ensuring that the dataset represents the broader tech industry rather than being limited to a static group of stocks. This dynamic selection helps create a dataset that can better generalize across different market conditions, improving the model's ability to make accurate predictions. Historical closing prices for these stocks are then gathered to create a time-series dataset.

The dataset is split into features and labels by date, and a sliding window with a shift size of 1 day is used to capture dynamic patterns in the stock market.

The training set consists of closing prices from earlier periods. After training, the model is validated on a separate set of closing prices from later periods, excluding the most recent data. Once optimal hyperparameters are identified, the model is retrained, this time including the validation set in the training process. Finally, the model is tested on the most recent closing prices. If the model's performance meets the desired criteria, it is retrained one last time, incorporating the test set, to make predictions on real-time data.

## Results

The model currently achieves an accuracy of 84.62%, falling short of the target accuracy of 95%. Several factors might contribute to this:

1. **Suboptimal Hyperparameter and Model Architecture**: Tuning these requires significant time and resources.
2. **Limited Dataset**: The dataset contains only 15 tech stocks, which may limit the model's ability to generalize effectively. However, it’s important to consider that increasing the number of stocks would significantly expand the dataset, potentially slowing down the training process. This trade-off between dataset size and training speed must be carefully balanced to optimize model performance.
3. **Inefficient Data Preprocessing**: The current method uses closing prices, but alternative approaches, such as using the direction of stock price changes, might yield better results.

Additionally, techniques like applying a moving average and differencing the time series data showed minimal performance improvements with the current approach. Although these methods were not effective, the code remains in the project, commented out in the stock_market_predictor/data/read_csv.py file under the remove_noise() function for future testing.

## Project Logistics

The prediction algorithm is stored under stock_market_predictor/.

- data/: Holds the time-series dataset and the Dataset class in PyTorch
- saved_models/: Stores all model variations and descriptions, saved using PyTorch
- config.py: Contains constants and hyperparameters
- data_preprocessing.py: Preprocess data and prepare DataLoaders in PyTorch
- model.py: Contains LSTM model
- train.py: Contains train script
- evaluate.py: Contains script to evaluate performance
- metrics.py: Contains metrics like accuracy and directional accuracy
- predict.py: _TODO_

_TODO_: test.py

performance.txt contains the current performance metrics of the model on the test set.

## How to Install and Run

Follow these steps to install and run the project:

1. Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/dhshin04/Parasite-Egg-Detection.git
```

2. Install necessary dependencies through the following command:

```bash
pip install -r requirements.txt
```

3. Import dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset).

4. To run train script, do the following:

```bash
cd stock_market_predictor
python train.py
```

5. To run the evaluation script, do the following:

```bash
cd stock_market_predictor
python evaluate.py
```

## How to Contribute

The goal of this project is to achieve a Mean Average Percent Error (MAPE) below 5%. Currently, the MAPE is above 10%, indicating room for improvement. Next steps could include further hyperparameter tuning, experimenting with new model architectures, and exploring alternative data preprocessing methods. If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## Credits

Author: Donghwa Shin [GitHub](https://github.com/dhshin04)

Kaggle Dataset used for training in APA format:
Oleh Onyshchak. (2020). Stock Market Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1054465

## Future Improvements

The project is still ongoing, as training requires a significant amount of time and financial resources (cost to run EC2 instance) due to the large dataset and the complexity of the LSTM neural network. Currently, the prediction algorithm achieves 85% accuracy, but the goal is to reach 95% or higher. Potential strategies to improve performance include:

1. Further hyperparameter tuning.
2. Modifying the model architecture.
3. Experimenting with new data preprocessing techniques, such as using the direction of stock price changes instead of raw closing prices.
