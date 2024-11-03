# region imports
from AlgorithmImports import *
from statsmodels.tsa.stattools import coint
import numpy as np
from scipy.stats import norm
# endregion

class FatFluorescentOrangeDinosaur(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(100000)
        self.SetBenchmark("SPY")

        # Define the pair of assets to trade
        self.symbol1 = self.AddEquity("AAPL", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("MSFT", Resolution.Daily).Symbol

        # Set lookback period for calculating z-score
        self.lookback = 20

        # Initialize a rolling window for price data
        self.price_window1 = RollingWindow[float](self.lookback)
        self.price_window2 = RollingWindow[float](self.lookback)

    def OnData(self, data: Slice):
        # Check if both symbols are in the slice
        if self.symbol1 not in data.Keys or self.symbol2 not in data.Keys:
            return

        # Update price data
        self.price_window1.Add(data[self.symbol1].Close)
        self.price_window2.Add(data[self.symbol2].Close)

        # Check if enough data is available
        if self.price_window1.Count < self.lookback:
            return

        # Calculate z-score
        z_score = self.calculate_z_score(self.price_window1, self.price_window2)

        # Check for entry/exit signals
        if z_score > 1:
            # Sell symbol1 and buy symbol2
            self.SetHoldings(self.symbol1, -0.5)
            self.SetHoldings(self.symbol2, 0.5)
        elif z_score < -1:
            # Buy symbol1 and sell symbol2
            self.SetHoldings(self.symbol1, 0.5)
            self.SetHoldings(self.symbol2, -0.5)
        else:
            # Clear positions
            self.Liquidate()

    def calculate_z_score(self, window1, window2):
        # Calculate spread between the two assets
        prices1 = np.array([i for i in window1])
        prices2 = np.array([i for i in window2])
        spread = np.log(prices1 / prices2)

        # Calculate mean and standard deviation of the spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Calculate z-score
        z_score = (spread[-1] - spread_mean) / spread_std

        return z_score