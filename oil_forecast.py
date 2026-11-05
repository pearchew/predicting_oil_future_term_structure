"""
Financial Econometrics group coursework: forecasting the oil futures term structure and CL1 returns.

Design (fastest defensible route):
  - Frequency: MONTHLY (every sheet already exists monthly, so no frequency mixing)
  - Target A (main):      next-month term structure  basis_{t+1} = (CL2 - CL1)/CL1   (contango > 0)
  - Target B (secondary): next-month CL1 log return   r_{t+1} = ln(CL1_{t+1}/CL1_t)
  - Predictor choice: correlation matrix on the TRAINING sample only
        keep |corr with target| >= 0.15, then drop the weaker of any pair with |corr| > 0.7
  - Models: naive benchmarks (random walk / zero, historical mean), AR(1), OLS on selected predictors,
            LASSO on all candidates (lambda by time-series CV)
  - Evaluation: expanding window, re-estimated monthly, out of sample from 2015-01
Run:  python oil_forecast.py   (Oil_Data.xlsx in the same folder). Outputs go to ./output
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (acorr_ljungbox, acorr_breusch_godfrey, het_white,
                                          het_breuschpagan, linear_reset, breaks_cusumolsresid)
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
OUT = "output"
os.makedirs(OUT, exist_ok=True)
FILE = "Oil_Data.xlsx"
OOS_START = pd.Period("2015-01", "M")
TRAIN_END = OOS_START - 1

# ========================================================================================
# STAGE 1: load + align (all monthly; month-end dates differ by a day -> match on year-month)
# ========================================================================================
def load(sheet):
    d = pd.read_excel(FILE, sheet_name=sheet).dropna(subset=["Date"])
    d["Date"] = pd.to_datetime(d["Date"]).dt.to_period("M")
    return d.set_index("Date").sort_index()

ind, mac, stk = load("Industry_M"), load("Macro_M"), load("StockPrices_M")
raw = ind.join(mac, how="inner").join(stk[["Gasoline All Grades", "Dow Jones US Oil&Gas Index"]], how="inner")

# ========================================================================================
# STAGE 2: stationary candidate predictors, all dated t (known at end of month t)
# ========================================================================================
lg = lambda s: np.log(s).diff()
X = pd.DataFrame(index=raw.index)
X["basis"]    = (raw["CL2"] - raw["CL1"]) / raw["CL1"]            # term structure: contango (+) / backwardation (-)
X["ret"]      = lg(raw["CL1"])                                    # CL1 return
X["d_crack"]  = raw["CRKS321C Index"].diff()                      # refining margin change
X["inv_g"]    = lg(raw["Inventories"])                            # inventory growth
X["rig_g"]    = lg(raw["Rig Count"])
X["prod_g"]   = lg(raw["Daily Production"])
X["net_comm"] = (raw["Commercial Long"] - raw["Commercial Short"]) / raw["Total Open Interest"]
X["bdi_g"]    = lg(raw["Baltic Dry Index"])                       # global demand proxy
X["dxy"]      = raw["DXY returns"]
X["spx_ret"]  = lg(raw["SPX"])
X["d_vix"]    = raw["VIX"].diff()
X["d_term"]   = (raw["10Y Yield"] - raw["3M Yield"]).diff()       # level fails ADF -> difference
X["og_ret"]   = lg(raw["Dow Jones US Oil&Gas Index"])
X["gas_g"]    = lg(raw["Gasoline All Grades"])
# CPI and IP are published ~2 weeks after month end -> extra lag to avoid look-ahead bias
X["ip_g"]     = lg(raw["IndustrialProduction Index"]).shift(1)
X["d_cpi"]    = raw["CPI YOY %"].diff().shift(1)
ALL = list(X.columns)

TARGETS = {
    # name: (series at t+1, own-lag column for AR(1), naive "no change" forecast column or None for zero)
    "TermStructure": (X["basis"].shift(-1), "basis", "basis"),
    "Return":        (X["ret"].shift(-1),   "ret",   None),
}

# ========================================================================================
# STAGE 3: stationarity (ADF)
# ========================================================================================
adf_rows = [("CL1 price (level)", *adfuller(raw["CL1"], autolag="AIC")[:2]),
            ("CL2-CL1 spread in $", *adfuller(raw["CL2"] - raw["CL1"], autolag="AIC")[:2]),
            ("10Y-3M spread (level)", *adfuller(raw["10Y Yield"] - raw["3M Yield"], autolag="AIC")[:2])]
adf_rows += [(c, *adfuller(X[c].dropna(), autolag="AIC")[:2]) for c in ALL]
adf = pd.DataFrame(adf_rows, columns=["series", "ADF stat", "p-value"]).round(3)
adf.to_csv(f"{OUT}/adf.csv", index=False)
print("ADF tests\n", adf.to_string(index=False))

# ========================================================================================
# helpers
# ========================================================================================
def select_by_correlation(train, target_col, thresh=0.15, max_pair=0.7):
    """Correlation-matrix screen on the training sample only."""
    c = train[[target_col] + ALL].corr()
    rank = c[target_col].drop(target_col).abs().sort_values(ascending=False)
    keep = []
    for v in rank[rank >= thresh].index:
        if all(abs(c.loc[v, k]) <= max_pair for k in keep):
            keep.append(v)
    return keep, c

def diagnostics(res, Xc, y):
    e = res.resid
    n, k = Xc.shape
    split = y.index.get_loc(pd.Period("2008-07", "M"))
    r1 = sm.OLS(y.iloc[:split], Xc.iloc[:split]).fit().ssr
    r2 = sm.OLS(y.iloc[split:], Xc.iloc[split:]).fit().ssr
    chowF = ((res.ssr - r1 - r2) / k) / ((r1 + r2) / (n - 2 * k))
    reset = linear_reset(res, power=3, use_f=True)
    d = {
        "Durbin-Watson": (durbin_watson(e), np.nan),
        "Breusch-Godfrey LM(6)": acorr_breusch_godfrey(res, nlags=6)[:2],
        "Ljung-Box Q(12)": tuple(acorr_ljungbox(e, lags=[12]).iloc[0]),
        "Breusch-Pagan": het_breuschpagan(e, Xc)[:2],
        "White": het_white(e, Xc)[:2],
        "ARCH: Ljung-Box on e^2 (12)": tuple(acorr_ljungbox(e**2, lags=[12]).iloc[0]),
        "Jarque-Bera": jarque_bera(e)[:2],
        "RESET (powers 2,3)": (reset.fvalue, reset.pvalue),
        "Chow break 2008-07": (chowF, 1 - stats.f.cdf(chowF, k, n - 2 * k)),
        "CUSUM": breaks_cusumolsresid(e, ddof=k)[:2],
    }
    return pd.DataFrame(d, index=["stat", "p-value"]).T.astype(float)

def pt_test(forecast_dir, actual_dir):
    """Pesaran-Timmermann (1992) test of directional accuracy. H0: forecasts have no directional skill.
    Inputs are booleans (True = up). Returns hit rate, PT statistic, one-sided p-value."""
    x, y = np.asarray(forecast_dir, bool), np.asarray(actual_dir, bool)
    n = len(x); P = np.mean(x == y); px, py = x.mean(), y.mean()
    Pstar = py * px + (1 - py) * (1 - px)
    vP = Pstar * (1 - Pstar) / n
    vPs = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n + 4 * px * py * (1 - px) * (1 - py) / n**2
    stat = (P - Pstar) / np.sqrt(vP - vPs) if vP > vPs else np.nan
    return P, stat, 1 - stats.norm.cdf(stat) if np.isfinite(stat) else np.nan

def dm_test(e1, e2):
    """Diebold-Mariano (h=1, squared loss). Negative stat => model 1 more accurate than model 2."""
    d = e1**2 - e2**2
    stat = d.mean() / np.sqrt(d.var(ddof=0) / len(d))
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))

# ========================================================================================
# STAGES 4-8 for each target
# ========================================================================================
summary = {}
for name, (yser, own, naive_col) in TARGETS.items():
    print("\n" + "=" * 90 + f"\nTARGET: {name}\n" + "=" * 90)
    df = pd.concat([yser.rename("y"), X], axis=1).dropna()
    train = df.loc[:TRAIN_END]

    # ---- STAGE 4: correlation matrix -> predictor selection (training sample only) ----
    CORE, corr = select_by_correlation(train, "y")
    if not CORE:                                   # nothing passes the screen -> fall back to economics
        CORE = ["basis", "inv_g", "net_comm", "dxy", "d_crack"]
    corr.round(2).to_csv(f"{OUT}/{name}_corr_matrix.csv")
    print("Corr with target (training):\n", corr["y"].drop("y").round(2).sort_values(key=abs, ascending=False).to_string())
    print("Selected for OLS:", CORE)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    plt.colorbar(im, fraction=.046); ax.set_title(f"{name}: correlation matrix, training sample 1995-2014 (y = next month)")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_fig0_corr_matrix.png", dpi=150); plt.close()

    # ---- STAGE 6: in-sample OLS + diagnostics (training sample) ----
    Xc = sm.add_constant(train[CORE])
    ols = sm.OLS(train["y"], Xc).fit()
    ols_nw = sm.OLS(train["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(ols_nw.summary())
    open(f"{OUT}/{name}_ols_summary.txt", "w").write(ols_nw.summary().as_text())
    diag = diagnostics(ols, Xc, train["y"]).round(3)
    vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=CORE).round(2)
    diag.to_csv(f"{OUT}/{name}_diagnostics.csv"); vif.to_csv(f"{OUT}/{name}_vif.csv")
    print("Diagnostics\n", diag.to_string(), "\nVIF\n", vif.to_string())

    # ---- STAGE 7: expanding-window out-of-sample forecasts ----
    naive_name = "Random walk (no change)" if naive_col else "Zero return"
    oos = df.loc[OOS_START:].index
    fc = pd.DataFrame(index=oos, columns=[naive_name, "Hist. mean", "AR(1)", "OLS selected", "LASSO"], dtype=float)
    picks = []
    for t in oos:
        tr, xt = df.loc[: t - 1], df.loc[[t]]
        fit_pred = lambda cols: sm.OLS(tr["y"], sm.add_constant(tr[cols])).fit().predict(
            sm.add_constant(xt[cols], has_constant="add")).iloc[0]
        fc.loc[t, naive_name] = xt[naive_col].iloc[0] if naive_col else 0.0
        fc.loc[t, "Hist. mean"] = tr["y"].mean()
        fc.loc[t, "AR(1)"] = fit_pred([own])
        fc.loc[t, "OLS selected"] = fit_pred(CORE)
        las = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), alphas=50, max_iter=20000)).fit(tr[ALL], tr["y"])
        fc.loc[t, "LASSO"] = las.predict(xt[ALL])[0]
        picks.append(pd.Series(las[-1].coef_ != 0, index=ALL))
    actual = df.loc[oos, "y"]
    current = df.loc[oos, naive_col] if naive_col else pd.Series(0.0, index=oos)   # value at t, for direction
    fc.to_csv(f"{OUT}/{name}_oos_forecasts.csv")
    sel = pd.concat(picks, axis=1).mean(axis=1).sort_values(ascending=False).round(2)
    sel.to_csv(f"{OUT}/{name}_lasso_selection_frequency.csv")
    print("LASSO selection frequency\n", sel.to_string())

    rows = {}
    for m in fc:
        err = actual - fc[m]
        base_err = actual - fc[naive_name]
        dm, p = dm_test(err.values, base_err.values) if m != naive_name else (np.nan, np.nan)
        rows[m] = {"RMSE": np.sqrt((err**2).mean()), "MAE": err.abs().mean(),
                   "OOS R2 vs naive": 1 - (err**2).sum() / (base_err**2).sum(),
                   "Direction hit rate": (np.sign(fc[m] - current) == np.sign(actual - current)).mean()
                   if m != naive_name else np.nan,
                   "DM stat vs naive": dm, "DM p": p}
        # robustness: drop 2020 (COVID storage crisis dominates squared errors)
        keep = ~oos.year.isin([2020])
        if m != naive_name:
            rows[m]["OOS R2 ex-2020"] = 1 - (err[keep]**2).sum() / (base_err[keep]**2).sum()
            rows[m]["DM p ex-2020"] = dm_test(err[keep].values, base_err[keep].values)[1]
    evalt = pd.DataFrame(rows).T.round(4)
    evalt.to_csv(f"{OUT}/{name}_oos_evaluation.csv")
    print("Out-of-sample evaluation\n", evalt.to_string())

    # ---- STAGE 8: trading rule ----
    # Return: long CL1 if forecast > 0 else short.  P&L = sign * next-month log return.
    # TermStructure: calendar spread. If spread expected to widen (more contango) -> long CL2 / short CL1.
    #   P&L proxy = sign(forecast change) * actual change in the $ spread (CL2 - CL1), per barrel.
    if name == "Return":
        pnl = pd.DataFrame({"Buy & hold": actual})
        for m in ["AR(1)", "OLS selected", "LASSO"]:
            pnl[m + " rule"] = np.sign(fc[m]) * actual
        unit = "log return"
    else:
        sp = (raw["CL2"] - raw["CL1"])
        dsp = (sp.shift(-1) - sp).loc[oos]
        pnl = pd.DataFrame({"Always long spread": dsp})
        for m in ["AR(1)", "OLS selected", "LASSO"]:
            pnl[m + " rule"] = np.sign(fc[m] - current) * dsp
        unit = "$/bbl"
    ann = pd.DataFrame({f"Avg monthly P&L ({unit})": pnl.mean(), "Monthly vol": pnl.std(),
                        "Ann. Sharpe": pnl.mean() / pnl.std() * np.sqrt(12),
                        "% months profitable": (pnl > 0).mean(),
                        "Max drawdown": pnl.cumsum().apply(lambda c: (c - c.cummax()).min())}).round(3)
    # ---- hit-rate tests ----
    #  direction: did the model call up/down correctly?  PT test (allows for the model leaning one way)
    #  profitable months: binomial test vs 50%
    hit = {}
    for m in ["AR(1)", "OLS selected", "LASSO"]:
        fdir, adir = (fc[m] - current) > 0, (actual - current) > 0
        P, pt, ptp = pt_test(fdir, adir)
        rule = pnl[m + " rule"]; rule = rule[rule != 0]
        wins = int((rule > 0).sum())
        hit[m] = {"Direction hit rate": P, "% forecasts 'up'": fdir.mean(), "PT stat": pt, "PT p (one-sided)": ptp,
                  "Profitable months": f"{wins}/{len(rule)}", "% profitable": wins / len(rule),
                  "Binomial p vs 50% (one-sided)": stats.binomtest(wins, len(rule), 0.5, alternative="greater").pvalue}
    first = pnl.columns[0]; base = pnl[first][pnl[first] != 0]
    hit[first] = {"Profitable months": f"{int((base > 0).sum())}/{len(base)}", "% profitable": (base > 0).mean(),
                  "Binomial p vs 50% (one-sided)": stats.binomtest(int((base > 0).sum()), len(base), 0.5, alternative="greater").pvalue}
    hitt = pd.DataFrame(hit).T.round(3)
    hitt.to_csv(f"{OUT}/{name}_hit_rate_tests.csv")
    ann.to_csv(f"{OUT}/{name}_trading_rule.csv")
    print("Trading rule\n", ann.to_string())
    print("Hit-rate tests\n", hitt.to_string())

    # ---- charts ----
    ax = pd.DataFrame({"Actual": actual, naive_name: fc[naive_name], "OLS selected": fc["OLS selected"]}).plot(figsize=(10, 3.5), alpha=.85)
    ax.set_title(f"{name}: actual vs forecast (out of sample)"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_fig2_oos_forecasts.png", dpi=150); plt.close()
    ax = pnl.cumsum().plot(figsize=(10, 3.5)); ax.set_title(f"{name}: cumulative P&L of trading rule ({unit})"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_fig3_trading_rule.png", dpi=150); plt.close()
    cum = ((actual.values[:, None] - fc.values) ** 2).cumsum(axis=0)
    cum = pd.DataFrame(cum, index=oos, columns=fc.columns)
    ax = (-(cum[["AR(1)", "OLS selected", "LASSO"]].sub(cum[naive_name], axis=0))).plot(figsize=(10, 3.5))
    ax.axhline(0, color="k", lw=.8); ax.set_title(f"{name}: cumulative squared-error advantage over {naive_name} (above 0 = better)"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_fig4_cum_sse.png", dpi=150); plt.close()
    summary[name] = evalt

# overview chart: price and term structure
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
raw["CL1"].plot(ax=ax[0], title="CL1 price, $/bbl (non-stationary)", color="#333"); ax[0].set_xlabel("")
(X["basis"] * 100).plot(ax=ax[1], title="Term structure (CL2-CL1)/CL1, % (stationary)", color="#1f6f8b")
ax[1].axhline(0, color="k", lw=.6); ax[1].set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_price_and_term_structure.png", dpi=150); plt.close()
print("\nDone. Files in ./output")
