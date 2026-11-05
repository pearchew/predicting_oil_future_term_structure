"""
Experiment: compare three feature sets built from the same raw monthly variables
  A) Level   x_t                (log level for price-like series)
  B) Delta   x_t - x_{t-1}      (first difference / growth)
  C) Gamma   Δx_t - Δx_{t-1}    (second difference / acceleration)
  D) All three together
Targets: next-month term structure (main) and next-month CL1 log return.
Same pipeline as oil_forecast.py: correlation screen on training data, OLS, LASSO, expanding window from 2015.
Run: python feature_sets.py [--anchor]  (--anchor: always include today's term structure level for the TS target)
      (needs Oil_Data.xlsx and oil_forecast.py in the same folder)
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
src = open("oil_forecast.py").read()
exec(src.split("# STAGE 3")[0])          # reuses load(), raw, X (for the targets), OOS_START, TRAIN_END
exec("def dm_test" + src.split("def dm_test")[1].split("# ====")[0])

# ---- raw monthly series (log for price-like, as-is for rates / ratios) ----
R = pd.DataFrame(index=raw.index)
R["basis"]    = (raw["CL2"] - raw["CL1"]) / raw["CL1"]
R["cl1"]      = np.log(raw["CL1"])
R["crack"]    = raw["CRKS321C Index"]
R["inv"]      = np.log(raw["Inventories"])
R["rig"]      = np.log(raw["Rig Count"])
R["prod"]     = np.log(raw["Daily Production"])
R["net_comm"] = (raw["Commercial Long"] - raw["Commercial Short"]) / raw["Total Open Interest"]
R["bdi"]      = np.log(raw["Baltic Dry Index"])
R["dxy"]      = raw["DXY returns"].cumsum()           # rebuild a log dollar-index level from its returns
R["spx"]      = np.log(raw["SPX"])
R["vix"]      = raw["VIX"]
R["term"]     = raw["10Y Yield"] - raw["3M Yield"]
R["og"]       = np.log(raw["Dow Jones US Oil&Gas Index"])
R["gas"]      = np.log(raw["Gasoline All Grades"])
R["ip"]       = np.log(raw["IndustrialProduction Index"]).shift(1)   # publication lag
R["cpi"]      = raw["CPI YOY %"].shift(1)                            # publication lag

LV = R.add_suffix("_L")
D1 = R.diff().add_suffix("_D")
D2 = R.diff().diff().add_suffix("_G")
SETS = {"A Level": LV, "B Delta": D1, "C Gamma": D2, "D All three": pd.concat([LV, D1, D2], axis=1)}

# stationarity share per set (training sample)
for nm, S in list(SETS.items())[:3]:
    p = [adfuller(S[c].loc[:TRAIN_END].dropna(), autolag="AIC")[1] for c in S]
    print(f"{nm}: {sum(np.array(p) < 0.05)}/{len(p)} series stationary at 5% (ADF)")

TARGETS = {"TermStructure": (X["basis"].shift(-1), R["basis"]), "Return": (X["ret"].shift(-1), None)}

def screen(train, cols, thresh=0.15, max_pair=0.7):
    c = train[["y"] + cols].corr()
    rank = c["y"].drop("y").abs().sort_values(ascending=False)
    keep = []
    for v in rank[rank >= thresh].index:
        if all(abs(c.loc[v, k]) <= max_pair for k in keep):
            keep.append(v)
    return keep

import sys
ANCHOR = "--anchor" in sys.argv
rows = []
for tname, (yser, now) in TARGETS.items():
    for sname, S in SETS.items():
        df = pd.concat([yser.rename("y"), S], axis=1).dropna()
        if now is not None:
            df["now"] = now
        cols = list(S.columns)
        sel = screen(df.loc[:TRAIN_END], cols)
        if ANCHOR and tname == "TermStructure":          # always give the model today's curve level
            if "basis_L" not in df: df["basis_L"] = R["basis"]
            if "basis_L" not in cols: cols = cols + ["basis_L"]
            if "basis_L" not in sel: sel = ["basis_L"] + sel
        oos = df.loc[OOS_START:].index
        f_ols, f_las, f_naive = [], [], []
        for t in oos:
            tr, xt = df.loc[: t - 1], df.loc[[t]]
            f_naive.append(xt["now"].iloc[0] if now is not None else 0.0)
            if sel:
                m = sm.OLS(tr["y"], sm.add_constant(tr[sel])).fit()
                f_ols.append(m.predict(sm.add_constant(xt[sel], has_constant="add")).iloc[0])
            else:
                f_ols.append(tr["y"].mean())
            las = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), alphas=30, max_iter=20000)).fit(tr[cols], tr["y"])
            f_las.append(las.predict(xt[cols])[0])
        a = df.loc[oos, "y"].values
        nv = np.array(f_naive)
        cur = df.loc[oos, "now"].values if now is not None else np.zeros(len(oos))
        keep = ~oos.year.isin([2020])
        for mname, f in [("OLS screened", np.array(f_ols)), ("LASSO", np.array(f_las))]:
            e, b = a - f, a - nv
            # trading: TS -> calendar spread $ P&L ; Return -> long/short CL1
            if tname == "TermStructure":
                sp = (raw["CL2"] - raw["CL1"])
                pnl = np.sign(f - cur) * (sp.shift(-1) - sp).loc[oos].values
            else:
                pnl = np.sign(f) * a
            rows.append({"Target": tname, "Feature set": sname, "Model": mname,
                         "# OLS vars": len(sel) if mname.startswith("OLS") else np.nan,
                         "OOS R2": 1 - (e**2).sum() / (b**2).sum(),
                         "OOS R2 ex-2020": 1 - (e[keep]**2).sum() / (b[keep]**2).sum(),
                         "DM p": dm_test(e, b)[1],
                         "Direction": np.mean(np.sign(f - cur) == np.sign(a - cur)),
                         "Sharpe": pnl.mean() / pnl.std() * np.sqrt(12)})
        print(f"{tname:13s} {sname:12s} OLS vars: {sel}")

res = pd.DataFrame(rows).round(3)
res.to_csv("output/feature_set_comparison%s.csv" % ("_anchored" if ANCHOR else ""), index=False)
pd.set_option("display.width", 200)
print("\n", res.to_string(index=False))
