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
usdt = df[(df.adv_asset == "USDT") & (df.adv_tradetype == "SELL")].copy()
usdt = usdt[usdt.timestamp > dt.datetime.now() - dt.timedelta(days=DAYS)]

# Estimate market price

vwap = {
    t: np.average(g.adv_price, weights=g.adv_tradablequantity)
    for t, g in usdt.groupby("timestamp")
}
usdt["vwap"] = usdt.timestamp.map(vwap)

# Rank users based on how their offers differ from the market price

usdt.groupby("advertiser_userno").apply(
    lambda _: (_.adv_price - _.vwap).sum(), include_groups=False
).sort_values().to_csv("data/rank.csv")

# Save all offers

table = Table.from_pandas(
    usdt[["advertiser_userno", "timestamp", "adv_price", "adv_tradablequantity"]]
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

with open("data/names.json", "w") as f:
    json.dump(usdt.groupby("advertiser_userno").advertiser_nickname.last().to_dict(), f)

pd.Series(vwap).to_csv("data/vwap.csv")
