# Stock Price Prediction with PyTorch (LSTM vs GRU)

This project aims to predict future stock prices using multivariate time series data (Open, Close, High, Low, Volume) with PyTorch. Two different Deep Learning architectures, LSTM and GRU, are implemented and their performance is compared.

## Project Steps
1. **Data Collection & Preparation:** Historical Apple (AAPL) stock data is fetched using the `yfinance` library. All features are scaled using `MinMaxScaler` to prevent gradient explosion.
2. **Sliding Window Implementation:** To train the sequential models, the data is transformed into 3D PyTorch tensors using a sliding window approach (60 days lookback period).
3. **Modeling:** Custom PyTorch models for LSTM and GRU are built from scratch, utilizing the Adam optimizer and Mean Squared Error (MSE) loss function.

## Results (LSTM vs GRU Comparison)
* **LSTM Model RMSE:** ~$7.69
* **GRU Model RMSE:** ~$5.39 (Winner)

**Conclusion:** The GRU model outperformed the LSTM on the test dataset. Due to its simpler architecture and fewer parameters, it trained significantly faster (approx. 80 seconds) and avoided overfitting.
