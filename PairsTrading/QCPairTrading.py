# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion

class EmotionalFluorescentPinkDinosaur(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        self.SetBenchmark("SPY")

        # Define the pairs of assets to trade
        self.tickers = ["KO", "PEP", "AAPL", "MSFT", "BAC", "JPM"]
        self.symbols = {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

        # Set lookback period for calculating z-score
        self.lookback = 20

        # Initialize a rolling window for price data
        self.price_windows = {}
        for symbol in self.symbols.values():
            self.price_windows[symbol] = RollingWindow[float](self.lookback)

    def OnData(self, data: Slice):
        # Ensure invalid pairs are removed.
        valid_symbols = []

        for i in range(0, len(self.tickers), 2):
            if (self.symbols[self.tickers[i]] in data.Keys and 
                self.symbols[self.tickers[i + 1]] in data.Keys):
                valid_symbols.append(self.symbols[self.tickers[i]])
                valid_symbols.append(self.symbols[self.tickers[i + 1]])

        # Update price data for all symbols
        for symbol in self.symbols.values():
            if symbol in data and data[symbol] != None:
                self.price_windows[symbol].Add(data[symbol].Close)

        # Calculate z-scores for all pairs
        z_scores = {}
        for i in range(0, len(self.tickers), 2):
            ticker1 = self.tickers[i]
            ticker2 = self.tickers[i + 1]
            symbol1 = self.symbols[ticker1]
            symbol2 = self.symbols[ticker2]

            if (symbol1 in valid_symbols and symbol2 in valid_symbols):
                z_scores[(ticker1, ticker2)] = self.calculate_z_score(self.price_windows[symbol1], self.price_windows[symbol2])

        self.Debug(f"Z-scores: {z_scores}")
        # Check for entry/exit signals based on z-scores
        for (ticker1, ticker2), z_score in z_scores.items():
            if z_scores[((ticker1, ticker2))] > 1:
                self.SetHoldings(self.symbols[ticker1], -1.5)
                self.SetHoldings(self.symbols[ticker2], 1.5)
                self.Debug(f"short {ticker1} long {ticker2} zscore {z_score}")
            elif z_scores[((ticker1, ticker2))] < -1:
                self.SetHoldings(self.symbols[ticker1], 1.5)
                self.SetHoldings(self.symbols[ticker2], -1.5)
                self.Debug(f"long {ticker1} short {ticker2} zscore {z_score}")
            else:
                self.Liquidate(ticker1)
                self.Liquidate(ticker2)
                self.Debug(f"Liquidating {ticker1}, {ticker2}")

    def calculate_z_score(self, window1, window2):
        # Calculate spread between the two assets
        prices1 = np.array([i for i in window1])
        prices2 = np.array([i for i in window2])
        spread = np.log(prices1 / prices2)

        # Calculate mean and standard deviation of the spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Calculate z-score
        z_score = (spread[-1] - spread_mean) / spread_std if spread_std != 0 else 0

        return z_score
    
    def calculate_z_threshold(self):
        # Calculate recent z scores
        recent_z_scores = []
        for symbol in self.symbols.values():
            if len(self.price_windows[symbol]) == self.lookback:
                for i in range(0, len(self.tickers), 2):
                    ticker1 = self.tickers[i]
                    ticker2 = self.tickers[i + 1]
                    z_score = self.calculate_z_score(self.price_windows[self.symbols[ticker1]], self.price_windows[self.symbols[ticker2]])
                    recent_z_scores.append(z_score)

        if (recent_z_scores != None):
            mean_z = np.mean(recent_z_scores)
            std_z = np.std(recent_z_scores)

            # 
            entry_threshold = mean_z + std_z
            exit_threshold = mean_z - std_z  

            return entry_threshold, exit_threshold
        
        return 1, -1 # default threshold.