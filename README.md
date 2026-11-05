# Oil term-structure forecasting — Financial Econometrics group coursework

Forecasts next month's WTI term structure, (CL2 - CL1) / CL1, with monthly data 1995-2024.
Training 1995-2014, out-of-sample test Jan 2015 - May 2024 (expanding window).

## Setup
    pip install pandas numpy statsmodels scikit-learn matplotlib openpyxl scipy
Put `Oil_Data.xlsx` in this folder (same folder as the scripts).

## Scripts (run in this order)
| Script | What it does | Time |
|---|---|---|
| `oil_forecast.py` | Main pipeline: data, ADF tests, correlation screen, OLS + diagnostics, AR(1), LASSO, out-of-sample tests, hit-rate tests, trading rule, charts | ~15 s |
| `feature_sets.py --anchor` | Robustness: levels vs first vs second differences | ~1 min |
| `rolling_window.py` | Robustness: expanding vs 20-year and 10-year rolling windows | ~15 s |
| `year_ranges.py` | Robustness: different training starts, split points, pre/post-2005 eras, volatility by period | ~10 s |

All results are written to `output/`.

## Key output files (term structure = main model)
| File | Contents |
|---|---|
| `adf.csv` | Stationarity tests |
| `TermStructure_corr_matrix.csv`, `TermStructure_fig0_corr_matrix.png` | Correlation matrix used to pick predictors |
| `TermStructure_ols_summary.txt` | OLS regression with Newey-West standard errors |
| `TermStructure_diagnostics.csv`, `TermStructure_vif.csv` | Diagnostic tests |
| `TermStructure_oos_forecasts.csv`, `TermStructure_oos_evaluation.csv` | Out-of-sample forecasts and accuracy (incl. without 2020) |
| `TermStructure_lasso_selection_frequency.csv` | How often LASSO picked each variable |
| `TermStructure_hit_rate_tests.csv` | Pesaran-Timmermann and binomial tests |
| `TermStructure_trading_rule.csv`, `TermStructure_fig3_trading_rule.png` | Calendar-spread trading results |
| `TermStructure_fig2_oos_forecasts.png`, `TermStructure_fig4_cum_sse.png` | Forecast charts |
| `fig1_price_and_term_structure.png` | Price vs term structure |
| `feature_set_comparison_anchored.csv` | Levels / deltas / gammas test |
| `window_comparison.csv`, `window_lasso_selection.csv` | Rolling-window test |
| `period_test_A.csv`, `_B.csv`, `_C.csv`, `period_volatility.csv`, `fig_volatility_regimes.png` | Sample-period tests |

`Return_*` files come from a secondary CL1-return model the main script still runs; the report does not use them.
