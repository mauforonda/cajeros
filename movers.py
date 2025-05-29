#!/usr/bin/env python

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import json
import datetime as dt
from pyarrow import parquet as pq
from pyarrow import Table

DAYS = 30


# Load data

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "andreschirinos/p2p-bob-exchange",
    "advice.parquet",
)
usdt = df[(df.asset == "USDT") & (df.tradetype == "SELL")].copy()
usdt = usdt[usdt.timestamp > dt.datetime.now() - dt.timedelta(days=DAYS)]

# Estimate market price

vwap = {
    t: np.average(g.price, weights=g.tradablequantity)
    for t, g in usdt.groupby("timestamp")
}
usdt["vwap"] = usdt.timestamp.map(vwap)

# Save all offers

table = Table.from_pandas(
    usdt[["advertiser_userno", "timestamp", "price", "tradablequantity"]]
)
pq.write_table(
    table,
    "data/ads.parquet",
    compression="zstd",
    compression_level=3,
    use_dictionary=True,
    data_page_version="2.0",
    write_statistics=True,
)

# Save additional information for graphics

pd.Series(vwap).to_csv("data/vwap.csv")

advertisers = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "andreschirinos/p2p-bob-exchange",
    "advertiser.parquet",
)

with open("data/names.json", "w") as f:
    json.dump(
        advertisers[advertisers.advertiser_userno.isin(usdt.advertiser_userno)]
        .set_index("advertiser_userno")
        .advertiser_nickname.to_dict(),
        f,
    )
