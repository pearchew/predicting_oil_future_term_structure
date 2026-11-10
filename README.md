# Forecasting the Oil Futures Term Structure

Financial Econometrics group coursework, Imperial College Business School.

We forecast **next month's change in the WTI crude oil calendar spread** and test whether the forecast is good enough to trade.

## The question in one paragraph

Crude oil is traded through futures contracts that deliver oil in a given month. **CL1** is the contract closest to delivery and **CL2** the next one. The gap between them, measured as a fraction of the front price,

```
spread = (CL2 − CL1) / CL1
```

describes the shape of the futures curve (the "term structure"). A positive spread is called **contango** (later oil costs more, usually when supply is plentiful); a negative spread is **backwardation** (oil now costs more, usually when supply is tight). The notebook asks: using information available at the end of a month, can we predict how this spread will move by the end of the next month, better than simply assuming it will not move?

## What is in the repository

| File | What it is |
|---|---|
| `Oil_Spread_Forecasting.ipynb` | The full analysis, from loading the data to the trading rule. Run top to bottom. |
| `Oil Data.xlsx` | The data. Only the three monthly sheets are used: `Industry_M` (CL1, CL2 and oil-industry series), `Macro_M` (macro series) and `StockPrices_M` (equity series). Daily and weekly sheets are not used. |

## Method

1. **Data (Steps 1 to 2).** Monthly data, Jan 1995 to Jun 2024. Individual oil-company stocks are removed (they duplicate the oil & gas index); Total Open Interest is removed (it equals Commercial Long + Commercial Short).
2. **Target (Step 2.3).** `y = spread(t+1) − spread(t)`, the change in the spread over the next month.
3. **Predictors (Step 4).** 17 market, industry and macro series plus today's spread. Trending series are converted to monthly percentage changes; CPI and industrial production get one extra month of lag because they are published late. Every predictor uses only information known at the time of the forecast.
4. **Stationarity checks (Step 5).** ADF and KPSS tests confirm the transformed series have a stable average and spread.
5. **Split (Step 6).** Training forecast origins Feb 1995 to Dec 2014 (239 months); test origins Jan 2015 to May 2024 (113 months).
6. **Benchmark (Step 6.2).** The random walk: next month's spread equals today's, so the forecast change is zero.
7. **Models (Step 7).** **LASSO**, a regression that switches off weak predictors automatically, with its penalty tuned by time-ordered cross-validation inside each training window. Plain **OLS** regression is the comparison. Both are refitted at each of the 113 test months using an **expanding** window (all past data) and a **rolling** window (the latest 120 months).
8. **Evaluation (Step 8).** RMSE, MAE, MAPE, out-of-sample R² against the random walk, and hit rate (share of months where the direction is right).
9. **Robustness (Step 9).** The same evaluation without the COVID months (Mar to May 2020, and all of 2020), both by removing them from the score and by re-running the model as if they never happened.
10. **Interpretation (Step 10).** Which predictors LASSO keeps, with what sign, and how stable that is over time.
11. **Diagnostics (Step 11).** Autocorrelation (Breusch-Godfrey, Ljung-Box), heteroskedasticity (Breusch-Pagan, ARCH-LM), parameter stability (Chow) and overlap between predictors (VIF).
12. **Trading (Step 12).** Go long the spread (buy CL2, sell CL1) when the forecast change is positive, short when negative. Profit is measured in dollars per barrel.

## Headline results (test period, 113 months)

| Window | Model | RMSE (pp) | Random-walk RMSE (pp) | Out-of-sample R² vs random walk | Hit rate |
|---|---|---|---|---|---|
| Expanding | LASSO | 2.34 | 2.45 | +9.0% | 60% |
| Expanding | OLS | 2.37 | 2.45 | +6.5% | 59% |
| Rolling | LASSO | 2.36 | 2.45 | +6.8% | 47% |
| Rolling | OLS | 2.68 | 2.45 | −19.6% | 55% |

- **LASSO with an expanding window is the main model.** It beats the random walk modestly, and the gain largely survives removing the COVID months (out-of-sample R² about +6% scored without them, about +10% when re-run without them).
- **Today's spread is always selected, with a negative sign:** when the spread is unusually high it tends to fall back (mean reversion). Gasoline prices, CPI, the 3-month yield, the oil & gas equity index and the rig count are selected some of the time.
- **Diagnostics flag problems:** autocorrelated and heteroskedastic errors, and structural breaks around 2005 and 2015. The relationship is not stable over time.
- **Trading:** the LASSO sign rule earned $16.05 per barrel of spread over the test period (Sharpe ratio 0.74), against −$1.65 for always holding the spread long. Without Mar to May 2020 this falls to $8.52 (Sharpe 0.52). Transaction costs and contract-roll effects are not included.

## Running it

Requires Python 3 with:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn scipy openpyxl jupyter
```

Then open `Oil_Spread_Forecasting.ipynb` and run all cells. The forecasting loops in Steps 7 and 9 refit LASSO hundreds of times and take a few minutes.

All project choices (dates, window length, cross-validation folds, COVID months, Chow break dates, transaction cost) are set in one cell, **Step 1.2**. Markers in the notebook show the status of each choice from the team's decision log: **DECIDED**, **PROPOSED** (placeholder, awaiting agreement) or **OPEN** (still to be discussed).

## Limitations

- The monthly CL1 and CL2 series switch to a new contract when the front month expires, so some measured spread changes include a roll effect a real position would not earn.
- Transaction costs are set to zero and position size is unconstrained.
- The spread's behaviour has changed over time, so past patterns may not continue.
- A few extreme months in 2020 have a large effect on the results.
- Diagnostic p-values are somewhat optimistic because the predictors were selected on the same data.
