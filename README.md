# DynamicHedgeBacktest
Allows to define a dynamic portfolio by rules of its greeks and backtest it on deribit historical data.

Obvious example is a delta-hedged option, but constant vega neutral vol flattener is implemented as example. 

Includes slippage, margin funding costs, perp funding...

## guide
1. run deribit_history
2. implement a Strategy class in deribit_portfolio.py
