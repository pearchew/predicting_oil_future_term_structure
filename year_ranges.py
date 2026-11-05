"""
Sample-period tests for the term-structure model.
  A) Training start date varies (test always Jan 2015 - May 2024, expanding window)
  B) Train/test split point varies (training always starts 1995, expanding window)
  C) Regime test: estimate ONCE on 1995-2004 only vs 2005-2014 only vs 1995-2014, then forecast 2015-2024
  Plus volatility by period and a rolling-volatility chart.
The correlation screen is re-run on each training sample (no look-ahead).
Run: python year_ranges.py   (needs Oil_Data.xlsx and oil_forecast.py in the same folder)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open("oil_forecast.py").read(); exec(src.split("# STAGE 3")[0])
for fn in ["select_by_correlation", "pt_test", "dm_test"]:
    exec("def " + fn + src.split("def " + fn)[1].split("\ndef ")[0].split("# ====")[0])
full = pd.concat([X["basis"].shift(-1).rename("y"), X], axis=1).dropna()
sp = raw["CL2"] - raw["CL1"]

# ---------- volatility by period ----------
vol = pd.DataFrame({"CL1 return vol (ann.)": X["ret"], "Term-structure change vol (pp/month)": X["basis"].diff() * 100})
periods = [("1995-2004", "1995", "2004"), ("2005-2014", "2005", "2014"), ("2015-2024", "2015", "2024"),
           ("2015-2024 ex-2020", None, None)]
vrows = []
for nm, a, b in periods:
    if a: v = vol.loc[a:b]
    else: v = vol.loc["2015":"2024"]; v = v[~v.index.year.isin([2020])]
    vrows.append({"Period": nm, "CL1 return vol (ann.)": v.iloc[:, 0].std() * np.sqrt(12),
                  "Term-structure monthly change, std (pp)": v.iloc[:, 1].std(),
                  "Avg term structure (%)": X["basis"].loc[v.index].mean() * 100,
                  "% months in contango": (X["basis"].loc[v.index] > 0).mean()})
V = pd.DataFrame(vrows).round(3); V.to_csv("output/period_volatility.csv", index=False)
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
(X["ret"].rolling(12).std() * np.sqrt(12) * 100).plot(ax=ax[0], color="#333")
ax[0].set_title("CL1 return volatility, rolling 12 months (annualised %)")
(X["basis"].diff().rolling(12).std() * 100).plot(ax=ax[1], color="#1f6f8b")
ax[1].set_title("Term-structure volatility: rolling 12-month std of monthly change (pp)")
for a_ in ax:
    for x in ["2005-01", "2015-01"]: a_.axvline(pd.Period(x, "M").ordinal, color="red", ls="--", lw=.8)
ax[1].set_xlabel("")
plt.tight_layout(); plt.savefig("output/fig_volatility_regimes.png", dpi=150); plt.close()

def evaluate(df, train_end, test_start, test_end=None, fixed=False):
    """Correlation screen on [start, train_end]; forecast test months with expanding (or fixed) estimation."""
    CORE, _ = select_by_correlation(df.loc[:train_end], "y")
    oos = df.loc[test_start:test_end].index
    a, cur = df.loc[oos, "y"], df.loc[oos, "basis"]
    dsp = (sp.shift(-1) - sp).loc[oos]; keep = ~oos.year.isin([2020])
    out = {}
    for nm, cols in [("AR(1)", ["basis"]), ("OLS", CORE)]:
        if fixed:
            m = sm.OLS(df.loc[:train_end, "y"], sm.add_constant(df.loc[:train_end, cols])).fit()
            f = m.predict(sm.add_constant(df.loc[oos, cols], has_constant="add"))
        else:
            f = pd.Series([sm.OLS(df.loc[:t - 1, "y"], sm.add_constant(df.loc[:t - 1, cols])).fit()
                           .predict(sm.add_constant(df.loc[[t], cols], has_constant="add")).iloc[0] for t in oos], index=oos)
        e, b = a - f, a - cur
        P, s, p = pt_test((f - cur) > 0, (a - cur) > 0)
        pnl = np.sign(f - cur) * dsp
        out[nm] = {"OOS R2": 1 - (e**2).sum() / (b**2).sum(),
                   "OOS R2 ex-2020": (1 - (e[keep]**2).sum() / (b[keep]**2).sum()) if keep.any() and (~keep).any() else np.nan,
                   "Direction": P, "PT p": p, "Sharpe": pnl.mean() / pnl.std() * np.sqrt(12)}
    return CORE, out

def row(label, n, CORE, out):
    r = {"Sample": label, "Train months": n, "OLS predictors": ", ".join(CORE)}
    for k, v in out["OLS"].items(): r["OLS " + k] = v
    r["AR(1) OOS R2"] = out["AR(1)"]["OOS R2"]; r["AR(1) OOS R2 ex-2020"] = out["AR(1)"]["OOS R2 ex-2020"]
    return r

# ---------- A: training start ----------
A = []
for start in ["1995-02", "1998-01", "2000-01", "2003-01", "2005-01", "2007-01", "2009-07"]:
    df = full.loc[start:]; CORE, out = evaluate(df, TRAIN_END, OOS_START)
    A.append(row(f"Train from {start[:4]}", len(df.loc[:TRAIN_END]), CORE, out))
A = pd.DataFrame(A)

# ---------- B: split point ----------
B = []
for split in ["2010-01", "2012-01", "2015-01", "2018-01"]:
    s = pd.Period(split, "M"); CORE, out = evaluate(full, s - 1, s)
    B.append(row(f"Test from {split[:4]}", len(full.loc[:s - 1]), CORE, out))
B = pd.DataFrame(B)

# ---------- C: regime (fixed estimation) ----------
C = []
for label, a_, b_ in [("1995-2004 only", "1995-02", "2004-12"), ("2005-2014 only", "2005-01", "2014-12"), ("1995-2014 (all)", "1995-02", "2014-12")]:
    df = pd.concat([full.loc[a_:b_], full.loc[OOS_START:]])
    CORE, out = evaluate(df, pd.Period(b_, "M"), OOS_START, fixed=True)
    C.append(row(label, len(full.loc[a_:b_]), CORE, out))
C = pd.DataFrame(C)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
for nm, T in [("A", A), ("B", B), ("C", C)]:
    T.round(3).to_csv(f"output/period_test_{nm}.csv", index=False); print(f"\n== {nm}\n", T.round(3).to_string(index=False))
print("\n== Volatility\n", V.to_string(index=False))
