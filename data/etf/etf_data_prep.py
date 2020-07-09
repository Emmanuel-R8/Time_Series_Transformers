################################################################################


# %%
import os

import numpy as np

import pandas as pd
import datetime as dt
import dateutil as du

# %%
# WARNING Make sure that the working directory is where you want the input files!!
dataDirETF = "."
dataDirIndicators = "."

# %%
# Load all the csv files as dataframes stored in a dict()
listSec = dict()

# We track the earliest date in all the time series used
minDate = dt.datetime(3000, 1, 1)
minSeries = ""

# WARNING check directory
for f in os.listdir(dataDirETF):

    # Assumes all the files are in CSV format
    if ".csv" in f:
        print(f)

        seriesName = f[0:-4]

        # Import the input and sets 'Date' as the index
        df = pd.read_csv(f"{dataDirETF}/{f}")
        df["Date"] = pd.to_datetime(df["Date"], yearfirst=True)
        df.set_index("Date")

        # Keep track of the earliest date
        if df["Date"][0] < minDate:
            minDate = df["Date"][0]
            minSeries = seriesName

        # Change the input to the changes day-to-day changes (in that order!):
        #    high/low  as log-change compared to the current day's close
        #    open/close as log-change compared to the previous day's close
        #    volume changed to log(volume)
        df["High"] = np.log(df["High"] / df["Open"])
        df["Low"] = np.log(df["Low"] / df["Open"])

        df["Open"] = np.log(df["Open"] / df["Close"].shift(1))
        df["Close"] = np.log(df["Close"] / df["Close"].shift(1))

        # Add 1 to avoid -Inf
        df["Volume"] = np.log(df["Volume"] + 1)

        # Remove Adj Close
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # rename the input
        df = df.rename(
            columns={
                "Open"  : f"{seriesName}.Open",
                "High"  : f"{seriesName}.High",
                "Low"   : f"{seriesName}.Low",
                "Close" : f"{seriesName}.Close",
                "Volume": f"{seriesName}.Volume",
            }
        )

        listSec[seriesName] = df

# %%
# Merge all the series from minDate
dataF = listSec[minSeries]

for k in listSec.keys():
    if k != minSeries:
        dataF = pd.merge(
            dataF, listSec[k], how="outer", on="Date", copy=False, sort=True
        )

# %%
dataF.to_csv("allData.csv", index=False)
dataF.to_pickle("allData.pickle")
