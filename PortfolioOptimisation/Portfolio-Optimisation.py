import sys
import matplotlib as mpl
import matplotlib.pyplot as plt #matplotlib for scatter graphs :)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt5.QtGui import QPalette, QColor

import pandas as pd
import numpy as np 
import seaborn as sns 
import yfinance as yf 
from scipy.optimize import minimize 
from datetime import date 

# =============================================================================================

#example portfolio with 6 assets, aims to determine what % of our portfolio
#we should invest in each ticker, based on data from 01/01/2023 - 01/01/2024
tickers = ['RR.L', 'NWG.L','HSBA.L', 'GSK.L', 'BARC.L', 'LLOY.L'] #.L is for london stock exchange
riskfree = 0.0106
dict_duration = {
    #date of form YYYY, MM, DD
    "start" : date(2023, 1, 1),
    "end" : date(2024, 1, 1),
}
data = yf.download(tickers, start=str(dict_duration["start"]), end=str(dict_duration["end"]))
data = data["Adj Close"] #Adjusted data at close (yfianance requires capital C!!)

#data.plot(figsize=(10, 5)) #plots the historical adjusted close prices

log_returns = np.log(data/ data.shift(1)) #data.shift shifts the rows by 1 so we can calulate the log returns
#log_returns[tickers].hist(figsize=[18, 12], bins=100)

#plot log returns as histogram
#plt.title("Histogram of Log Returns")
#plt.xlabel("Log Returns")
#plt.ylabel("Frequency")
#plt.show()

def portfolio_performance(weights, log_returns): #weights, how much of each asset we'll put into each stock
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    portfolio_returns = np.sum(mean_returns * weights) * 252 #252 trading days in a year
    # = w^T Sigma w
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return portfolio_returns, portfolio_volatility

#the sharpe ratio is 
def Portfolio_sharpe(weights, log_returns, riskfree):
    p_returns, p_volatility = portfolio_performance(weights, log_returns)
    return -(p_returns - riskfree) / p_volatility

#==============================================================================================
#Optimisation methodologies:
#Method I - Monte-Carlo simulation

def weights_MonteCarlo():
    weights = np.random.random(len(tickers))
    return (weights / np.sum(weights)) #normalise the weights so |w| =< 1
    

#==============================================================================================
#generate data for 10k portfolios
num_portfolios = 10_000 #70k portfolios takes like 10 years on my laptop
all_weights = np.zeros((num_portfolios, len(tickers)))
returns = np.zeros(num_portfolios)
volatilities = np.zeros(num_portfolios)
sharpe_ratios = np.zeros(num_portfolios)
#initialise variables for optimal portfolio
maxSharpe = -99999
maxWeights = np.random.random(len(tickers))
maxReturns = 0
maxVol = 0

for i in range(num_portfolios):
    #generate weightings for each ticker in portfolio
    weights = weights_MonteCarlo()
    
    all_weights[i] = weights
    returns[i], volatilities[i] = portfolio_performance(weights, log_returns)
    sharpe_ratios[i] = -(Portfolio_sharpe(weights, log_returns, riskfree)) #negative because "some maths"
    
    #store the portfolio which delivered the best sharpe value
    if (maxSharpe < sharpe_ratios[i]):
        maxSharpe = sharpe_ratios[i]
        maxWeights = all_weights[i]
        maxReturns = returns[i]
        maxVol = volatilities[i]

#plot data as scatter graph using matplotlib

#display optimal portfolio, bar chart with weights of each ticker.
plt.figure(figure=(10, 6))
plt.bar(tickers, maxWeights * 100, color='Black', width=0.4)
plt.xlabel("Stock ticker")
plt.ylabel("Proportion of portfolio (%)")
plt.title(f"Optimal portfolio with sharpe ratio {maxSharpe.round(2)}")
plt.show()

#display optimal portfolio as a donut:
fig, ax = plt.subplots()
ax.pie(maxWeights, labels=tickers, autopct='%1.1f%%')
plt.show()

# ========================================================================

# GUI implementation:

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Portfolio Optimiser")

        width = 800
        height = 1_000

        self.setMaximumSize(width, height)
        self.setMinimumSize(width, height)

        layout = QGridLayout()

        sharpeScatter = MplWidget()

        # add sharpe scatter to layout
        layout.addWidget(sharpeScatter, 0, 0)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(MplCanvas, self).__init__(Figure(figsize=(width, height), dpi=dpi))
        self.axes = self.figure.add_subplot(111)
        self.setParent(parent)

class MplWidget(QWidget):
    def __init__ (self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = MplCanvas()
        self.canvas.axes.scatter(
            returns * 100,
            volatilities,
            c=sharpe_ratios,
            cmap="viridis"
        )
        self.canvas.axes.xaxis = "Volatility"
        self.canvas.axes.yaxis = "Expected returns (%)"
        

        plt.figure(figure=(10, 6))
        scatter = plt.scatter(volatilities, returns * 100, c=sharpe_ratios, cmap="viridis")
        plt.colorbar(scatter, label='Sharpe ratio')
        plt.xlabel("Volatility (risk)")
        plt.ylabel("Expected returns (%)")
        plt.title("Feasible set")
        plt.plot(maxVol, maxReturns * 100, markersize=10, markerfacecolor="red")
        plt.show()

        
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
