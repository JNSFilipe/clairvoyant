import pandas as pd

# Local imports
import sys

sys.path.append("../.")
from clairvoyant.engine import Backtest

features  = ["X1", "X2"]     # Financial indicators of choice
trainStart = 0               # Start of training period
trainEnd   = 200             # End of training period
testStart  = 201             # Start of testing period
testEnd    = 321             # End of testing period
buyThreshold  = 0.65         # Confidence threshold for predicting buy (default = 0.65) 
sellThreshold = 0.65         # Confidence threshold for predicting sell (default = 0.65)
continuedTraining = False    # Continue training during testing period? (default = false)

# Initialize backtester
backtest = Backtest(features, trainStart, trainEnd, testStart, testEnd, buyThreshold, sellThreshold, continuedTraining)

# A little bit of pre-processing
data = pd.read_csv("data.csv", date_parser=['date'])
data = data.round(3)                                    

# Start backtesting and optionally modify SVC parameters
# Available paramaters can be found at: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
backtest.start(data, kernel='rbf', C=1, gamma=10)
backtest.conditions()
backtest.statistics()  
backtest.visualize('Example')
