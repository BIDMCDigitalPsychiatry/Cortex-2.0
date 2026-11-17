"""Cortex-2.0 mobility sensitivity and reference-validation statistics.

All comparisons are paired within participant/day. This module computes
statistics; it does not automatically declare a tested algorithm superior.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment


def absolute_agreement_icc(values: np.ndarray):
    """ICC(A,1), two-way absolute agreement across K parameter settings.

    Rows are participant-days, columns are settings. Only complete rows used.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 2:
        raise ValueError("ICC requires a two-dimensional matrix")
    a = a[np.isfinite(a).all(axis=1)]
    n, k = a.shape
    if n < 2 or k < 2:
        return np.nan
    grand = a.mean()
    row_means = a.mean(axis=1)
    col_means = a.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand)**2)
    ss_cols = n * np.sum((col_means - grand)**2)
    ss_error = np.sum((a - row_means[:,None] - col_means[None,:] + grand)**2)
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else np.nan


def feature_stability(grid_results, *,
                      unit_cols=("participant_id", "date"),
                      setting_col="setting", features=("path_length_m", "displacement_m")):
    """Report absolute-agreement ICC and rank preservation across settings."""
    rows=[]
    for feature in features:
        wide = grid_results.pivot_table(index=list(unit_cols),columns=setting_col,values=feature,aggfunc="first")
        wide = wide.dropna()
        icc=absolute_agreement_icc(wide.to_numpy())
        ranks=[]
        for i in range(len(wide.columns)):
            for j in range(i+1,len(wide.columns)):
                if len(wide)>=3:
                    coefficient=spearmanr(wide.iloc[:,i],wide.iloc[:,j]).statistic
                    if np.isfinite(coefficient): ranks.append(float(coefficient))
        rows.append({"feature":feature,"n_units":len(wide),"n_settings":len(wide.columns),
                     "icc_absolute_agreement":icc,
                     "mean_pairwise_spearman":np.mean(ranks) if ranks else np.nan,
                     "min_pairwise_spearman":np.min(ranks) if ranks else np.nan})
    return pd.DataFrame(rows)


def association_stability(grid_results, *, outcome_col, feature,
                          setting_col="setting", unit_cols=("participant_id", "date")):
    """Descriptive Spearman feature/outcome association under each setting.

    Clinical inference, repeated measurements, and mixed models must be
    handled separately rather than interpreting these p-values as independent.
    """
    df=grid_results.drop_duplicates(list(unit_cols)+[setting_col])
    rows=[]
    for setting, subset in df.groupby(setting_col):
        x=pd.to_numeric(subset[feature],errors="coerce")
        y=pd.to_numeric(subset[outcome_col],errors="coerce")
        valid=x.notna()&y.notna()
        r=spearmanr(x[valid],y[valid]).statistic if valid.sum()>=3 else np.nan
        rows.append({"setting":setting,"feature":feature,"outcome":outcome_col,
                     "n":int(valid.sum()),"spearman_r":float(r)})
    return pd.DataFrame(rows)


def paired_reference_metrics(comparisons, *, reference_col, estimator_cols,
                             unit_cols=("participant_id","date")):
    """Full-day paired MAE, RMSE, bias, APE; estimates compared on common rows."""
    cols=[reference_col]+list(estimator_cols)
    frame=comparisons.dropna(subset=cols).copy()
    ref=pd.to_numeric(frame[reference_col],errors="coerce").to_numpy(float)
    rows=[]
    for estimator in estimator_cols:
        est=pd.to_numeric(frame[estimator],errors="coerce").to_numpy(float)
        valid=np.isfinite(ref)&np.isfinite(est)
        diff=est[valid]-ref[valid]
        r=ref[valid]
        nonzero=np.abs(r)>1e-9
        rows.append({"estimator":estimator,"n":int(valid.sum()),
                     "n_units":int(frame.loc[valid,list(unit_cols)].drop_duplicates().shape[0]),
                     "mae":float(np.mean(np.abs(diff))) if len(diff) else np.nan,
                     "rmse":float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan,
                     "bias":float(np.mean(diff)) if len(diff) else np.nan,
                     "mape_pct":float(np.mean(100*np.abs(diff[nonzero]/r[nonzero]))) if nonzero.any() else np.nan})
    return pd.DataFrame(rows)


def interval_coverage(rows, *, reference_col, lower_col, upper_col):
    """Empirical reference coverage. Only call with genuinely valid intervals."""
    d=rows.dropna(subset=[reference_col,lower_col,upper_col]);n=len(d)
    return {"n":n,"coverage_fraction":float(((d[lower_col]<=d[reference_col]) &
             (d[reference_col]<=d[upper_col])).mean()) if n else np.nan}


def match_trips(predicted, reference, *, start_col="trip_start_ms",
                end_col="trip_end_ms", tolerance_ms=600_000):
    """One-to-one trip matching by departure and arrival within tolerance."""
    p=predicted[[start_col,end_col]].to_numpy(float)
    r=reference[[start_col,end_col]].to_numpy(float)
    if not len(p) or not len(r):
        tp=0
    else:
        cost=(np.abs(p[:,None,0]-r[None,:,0]) +
              np.abs(p[:,None,1]-r[None,:,1]))
        valid=(np.abs(p[:,None,0]-r[None,:,0])<=tolerance_ms)&(
              np.abs(p[:,None,1]-r[None,:,1])<=tolerance_ms)
        penalty=1e16
        ix,jx=linear_sum_assignment(np.where(valid,cost,penalty))
        tp=int(valid[ix,jx].sum())
    precision=tp/len(p) if len(p) else (1.0 if not len(r) else 0.0)
    recall=tp/len(r) if len(r) else (1.0 if not len(p) else 0.0)
    return {"tp":tp,"fp":len(p)-tp,"fn":len(r)-tp,
            "precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall>0 else 0.0}


def run_sensitivity_grid(windows, settings, recompute, *,
                         participant_col="participant_id", date_col="date"):
    """Re-run the same feature callback for pre-specified setting combinations.

    windows: iterable of dictionaries containing participant_id, date, fixes,
    start_ms, end_ms, and optional outcome values. recompute is the SAME
    callable used for production estimates; no alternative segmentation here.
    """
    out=[]
    for window in windows:
        for name, parameter_values in settings.items():
            features=recompute(
                window["fixes"], start_ms=window["start_ms"],
                end_ms=window["end_ms"], **parameter_values)
            row={participant_col:window[participant_col], date_col:window[date_col],
                 "setting":name,**features}
            for key, value in window.items():
                if key not in {"fixes","start_ms","end_ms",participant_col,date_col}:
                    row[key]=value
            out.append(row)
    return pd.DataFrame(out)
