# ideas from online resources to improve current engine

1. run montecarlo were it randomly remove trades from the backtest. Also removing the 1 percentile of bad and good trades to find good parameters is also important, such that the strategy doesn't rely on a few outliers. Also, when developing algos for small timeframe where intrabar order execution happens, its important to have good tickdata, if not the backtesting program would guess that you would get executed but in reality there was no transaction at that price, not important for higher timeframe buying at the opening price though. the higher the timeframe, the easier it is to create profitable algos, however fewer trades = less statistically significant and longer time to create profit

2. for any scalping strategy on fast charts you MUST use tick data and a backtesting engine that replicates real spreads and slippage, at least at top of book. Otherwise your results might be wildly optimistic.
 

3. https://www.youtube.com/watch?v=W722Ca8tS7g 
    3.1 perameter sensitivity - to test for overfitting & find rebust and consistent performing strategy. b/c market is changing
    3.2 walk forward optimisation/ validation. (we don't want to torture the data and try every combination to find the working combination, we want a robust model that work on out-of-sample data)
    3.3 stress testing - test during 2008 financial crisis, 2020 covid crash, slippage, flash crash, **execution delays** etc. Some common examples: double commision, triple slippage, add random execution delays.
    3.4 Monte-carlo stimulation: to see true risk range(not just one historical path/ realistic expectations; survivability)

4. https://www.youtube.com/watch?v=mkzcntzznMc
    4.1 Latency handling and scalabilty
    4.2 Sharpe: higher sharpe minimize draw-downs, use leverage safety and stable returns
    4.3 simple returns' limitatation: it is not symetric so we need to use log returns which has the property of time compoundability
    4.4 regular/irrelgular time series: time is arbitary b/w data points 
    4.5 how does trader overcome gradient decent - train-inference mismatch for auto-regressive model??
    4.6 Occam’s Razor: If the complex version is only slightly better, prefer the simpler one unless the extra complexity has a strong reason.

     


5. public strategy are widely known. If use them naively, the edge may be weak or gone. Own contribution can focus on:
        Better universe selection
        Better risk control
        Better transaction cost model
        Better portfolio construction
        Better regime filter
        Better validation
        Better execution assumptions
        Combining multiple weak signals


next step:
A. The missing engine (do this first — it's the actual backtest)

A portfolio/PnL simulator: positions × next-period returns − costs → an equity curve. Right now you compute positions and costs but never combine them into strategy returns. This is the single biggest gap.
Then performance metrics: Sharpe, Sortino, max drawdown, CAGR, hit rate, turnover.
B. Statistical rigor ("deeper statistical controls") — this is what separates a strategy that looks good from one that is good:

Walk-forward / out-of-sample validation — never tune and test on the same data (your README lists this as a goal).
Multiple-testing / overfitting controls — if you try 100 lookback windows and pick the best, that "best" is mostly luck. Learn: deflated Sharpe ratio, the "backtest overfitting" literature (Bailey & López de Prado), purged k-fold cross-validation with embargo.
Statistical significance of returns, not just point estimates: bootstrap confidence intervals, regime testing.
C. Realistic data & execution ("huge data pipelines"):

Survivorship bias — yfinance only gives you companies that still exist. A real backtest needs delisted tickers, or it overstates returns badly.
Point-in-time data — knowing what was actually known on each date (no revised fundamentals leaking back).
A storage/caching layer — Parquet/DuckDB/a database instead of re-fetching from yfinance every run. This is the "pipeline" part: ingest → clean → store → serve.
Better cost/slippage modeling — volume-aware impact, not a hand-picked coefficient.
D. "Multiple layers" usually means moving beyond one signal: factor models (combining momentum, value, mean-reversion), risk models (covariance, position sizing via volatility targeting or Kelly), and proper portfolio construction (optimization with constraints) rather than naive +1/−1.




