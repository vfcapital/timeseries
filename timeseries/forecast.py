import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def find_least_bitcoin_correl(coin_returns):
    # find coin with least bitcoin correlation:
    bitcoin_correl = coin_returns.dropna().corr()["bitcoin"]
    bitcoin_correl_2 = pd.DataFrame(bitcoin_correl * bitcoin_correl)
    index = bitcoin_correl_2[
        bitcoin_correl_2["bitcoin"] == bitcoin_correl_2["bitcoin"].min()
    ].index
    series = coin_returns[index].dropna()
    return series


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.values
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype("float32")
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print("ARIMA%s RMSE=%.3f" % (order, rmse))
                except:
                    continue
    print("Best ARIMA%s RMSE=%.3f" % (best_cfg, best_score))


def analyse_final_model(series, arima_order):
    X = series.values
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    plt.plot(test)
    plt.plot(predictions, color="red")
    plt.title(
        'Coin: "'
        + str(series.columns[0])
        + '" - Forecasting ARIMA'
        + str(arima_order)
    )
    plt.grid()
    plt.show()
    print(model_fit.summary())
