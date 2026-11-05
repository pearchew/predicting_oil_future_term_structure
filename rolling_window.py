"""
Robustness: expanding vs rolling (sliding) estimation windows for the term-structure models.
  expanding  : train on all data from 1995 up to month t-1
  rolling 240: train on the most recent 240 months (20 years, same length as the original training sample)
  rolling 120: train on the most recent 120 months (10 years)
Reports out-of-sample accuracy, direction tests, trading P&L and how often LASSO selects each variable.
Run: python rolling_window.py   (needs Oil_Data.xlsx and oil_forecast.py in the same folder)
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
src = open("oil_forecast.py").read()
exec(src.split("# STAGE 3")[0])
for fn in ["select_by_correlation", "pt_test", "dm_test"]:
    exec("def " + fn + src.split("def " + fn)[1].split("\ndef ")[0].split("# ====")[0])

df = pd.concat([X["basis"].shift(-1).rename("y"), X], axis=1).dropna()
CORE, _ = select_by_correlation(df.loc[:TRAIN_END], "y")
oos = df.loc[OOS_START:].index
actual, cur = df.loc[oos, "y"], df.loc[oos, "basis"]
sp = raw["CL2"] - raw["CL1"]; dsp = (sp.shift(-1) - sp).loc[oos]
keep = ~oos.year.isin([2020])

res, sel = [], {}
for scheme, W in [("Expanding", None), ("Rolling 240m", 240), ("Rolling 120m", 120)]:
    fc = {"AR(1)": [], "OLS selected": [], "LASSO": []}; picks = []
    for t in oos:
        tr = df.loc[: t - 1]
        if W: tr = tr.iloc[-W:]
        xt = df.loc[[t]]
        for nm, cols in [("AR(1)", ["basis"]), ("OLS selected", CORE)]:
            fc[nm].append(sm.OLS(tr["y"], sm.add_constant(tr[cols])).fit().predict(sm.add_constant(xt[cols], has_constant="add")).iloc[0])
        las = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), alphas=50, max_iter=20000)).fit(tr[ALL], tr["y"])
        fc["LASSO"].append(las.predict(xt[ALL])[0]); picks.append(pd.Series(las[-1].coef_ != 0, index=ALL))
    P = pd.concat(picks, axis=1).T
    sel[scheme] = P.mean().round(2)
    for nm, f in fc.items():
        f = pd.Series(f, index=oos); e, b = actual - f, actual - cur
        hit, pt, ptp = pt_test((f - cur) > 0, (actual - cur) > 0)
        pnl = np.sign(f - cur) * dsp
        res.append({"Window": scheme, "Model": nm,
                    "OOS R2": 1 - (e**2).sum() / (b**2).sum(),
                    "OOS R2 ex-2020": 1 - (e[keep]**2).sum() / (b[keep]**2).sum(),
                    "Direction": hit, "PT p": ptp,
                    "Spread Sharpe": pnl.mean() / pnl.std() * np.sqrt(12),
                    "Avg # LASSO vars": P.sum(axis=1).mean() if nm == "LASSO" else np.nan,
                    "# distinct LASSO sets": P.astype(int).astype(str).agg("".join, axis=1).nunique() if nm == "LASSO" else np.nan})
R = pd.DataFrame(res).round(3); S = pd.DataFrame(sel)
R.to_csv("output/window_comparison.csv", index=False); S.to_csv("output/window_lasso_selection.csv")
pd.set_option("display.width", 200)
print(R.to_string(index=False)); print("\nLASSO selection frequency\n", S[(S > 0).any(axis=1)].to_string())
