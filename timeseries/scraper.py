from datetime import datetime
from io import BytesIO

import pandas as pd
import requests


def parse_dates(month, year):
    start_dt = int(datetime(year, month, 1, 0, 0).timestamp())
    end_dt = int(datetime(year, month + 1, 1, 0, 0).timestamp())
    return start_dt, end_dt


def get_gecko_data(coin):
    headers = {
        "authority": "www.coingecko.com",
        "method": "GET",
        "path": "/price_charts/export/1/usd.csv",
        "scheme": "https",
        "accept": (
            "text/html,application/xhtml+xml,application/"
            "xml;q=0.9,image/avif,image/webp,image/apng,*"
            "/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
        ),
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "if-none-match": 'W/"2fa3f8c23337552cfc3fc04218f09172"',
        "referer": (
            "https://www.coingecko.com/de/munze/"
            + coin
            + "/historical_data?start_date=2021-05-01&end_date=2022-06-30"
        ),
        "sec-ch-ua": (
            '".Not/A)Brand";v="99", '
            '"Google Chrome";v="103", '
            '"Chromium";v="103"'
        ),
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            "AppleWebKit/537.36 (KHTML, like Gecko)"
            "Chrome/103.0.0.0 Safari/537.36"
        ),
    }
    params = {}
    response = requests.get(
        "https://www.coingecko.com/price_charts/export/" + coin + "/usd.csv",
        params=params,
        headers=headers,
    )
    return response


def get_coin_df(coin):
    data = get_gecko_data(coin)
    df = pd.read_csv(BytesIO(data.content))
    df = (
        df.drop(["market_cap", "total_volume"], axis=1)
        .rename(columns={"snapped_at": "date", "price": coin})
        .set_index("date")
    )
    df.index = pd.to_datetime(df.index.str[:10])
    return df


def get_coin_list():
    response = requests.get(
        "https://api.coingecko.com/api/v3/coins/"
        "markets?vs_currency=usd&order=market_cap_desc"
    )
    coin_list = []
    raw_coin_id_list = response.json()
    i = 0
    for coin in response.json():
        while i < 50:
            coin_list.append(raw_coin_id_list[i]["id"])
            i = i + 1
    return coin_list


def get_coin_prices():
    df_list = []
    coin_list = get_coin_list()
    for coin in coin_list:
        df_list.append(get_coin_df(coin))
    return pd.concat(df_list, axis=1)


def get_coin_returns():
    return get_coin_prices().pct_change()
