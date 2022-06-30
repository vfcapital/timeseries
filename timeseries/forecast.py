from scraper import get_coin_returns
import pandas as pd
from matplotlib import pyplot


def find_least_bitcoin_correl(coin_returns):
    # find coin with least bitcoin correlation:
    bitcoin_correl = coin_returns.dropna().corr()["bitcoin"]
    bitcoin_correl_2 = pd.DataFrame(bitcoin_correl * bitcoin_correl)
    index = bitcoin_correl_2[
        bitcoin_correl_2["bitcoin"] == bitcoin_correl_2["bitcoin"].min()
    ].index
    series = coin_returns[index].dropna()
    return series
