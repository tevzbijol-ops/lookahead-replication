"""
Step 2 of the Build: replicate LookAhead's XGBoost detector and test an
unsupervised Isolation Forest that never sees a labelled attack.

    pip install pandas scikit-learn xgboost matplotlib
    python3 build.py lookahead_features.csv

Split = the paper's last fold (data_loader.getCrossDataSet, fold 4):
each class is sorted oldest -> newest, the oldest 80 % trains, the newest
20 % tests. Training only on the past is what makes it a fair test of
"would this have caught the next attack?".

Outputs: results.md (table), pr_curve.png (Figure 1), confusion matrices.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from xgboost import XGBClassifier

SEED = 42
FUND_LABELS = [" Anonymous", "Safe", "unknown_low", "Bridge", "unknown_contract", "unknown_high"]
DROP = ["contract_address", "chain", "order_key", "tag", "isTokenContract", "isERC1967Proxy"]


def to_int(x):
    s = str(x)
    return int(s, 16) if s.startswith("0x") else int(float(s))


def preprocess(df):
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    X["gasUsed"] = X["gasUsed"].map(to_int)
    X["nonce"] = X["nonce"].map(to_int)
    X["value"] = (X["value"].map(to_int) > 0).astype(int)
    X["verify_tag"] = X["verify_tag"].astype(str).map({"True": 1, "False": 0}).fillna(0)
    X["fund_from_label"] = pd.Categorical(X["fund_from_label"], categories=FUND_LABELS)
    X = pd.get_dummies(X, columns=["fund_from_label"], prefix="fund")
    return X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)


def chrono_split(df, train_frac=0.8):
    parts_tr, parts_te = [], []
    for tag in (0, 1):
        d = df[df.tag == tag].sort_values("order_key").reset_index(drop=True)
        cut = int(len(d) * train_frac)
        parts_tr.append(d.iloc[:cut]); parts_te.append(d.iloc[cut:])
    return pd.concat(parts_tr, ignore_index=True), pd.concat(parts_te, ignore_index=True)


def metrics(name, y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"model": name,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "FPR %": 100 * fp / (fp + tn),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def main(path):
    df = pd.read_csv(path, dtype=str)
    df["tag"] = df["tag"].astype(int)
    df["order_key"] = df["order_key"].astype(float)
    train, test = chrono_split(df)
    Xtr, Xte = preprocess(train), preprocess(test)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    ytr, yte = train.tag.values, test.tag.values

    scaler = StandardScaler().fit(Xtr)
    Str, Ste = scaler.transform(Xtr), scaler.transform(Xte)

    lines = [f"train: {len(train)} contracts ({ytr.sum()} attacks) | "
             f"test: {len(test)} contracts ({yte.sum()} attacks)"]

    # 1) Replication: supervised XGBoost (their first-level model)
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED)
    xgb.fit(Str, ytr)
    p_xgb = xgb.predict_proba(Ste)[:, 1]
    rows = [metrics("XGBoost (supervised, replication)", yte, (p_xgb >= 0.5).astype(int))]

    # 2) Experiment: Isolation Forest trained on benign contracts only
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)
    iso.fit(Str[ytr == 0])
    s_tr_benign = -iso.score_samples(Str[ytr == 0])
    s_te = -iso.score_samples(Ste)
    for pct in (99.0, 99.5):  # threshold set from clean (benign) training data only
        thr = np.percentile(s_tr_benign, pct)
        rows.append(metrics(f"Isolation Forest, threshold p{pct}", yte, (s_te >= thr).astype(int)))

    res = pd.DataFrame(rows)
    ap_x, ap_i = average_precision_score(yte, p_xgb), average_precision_score(yte, s_te)
    lines += ["", res.to_markdown(index=False, floatfmt=".3f"), "",
              f"Average precision (area under PR curve): XGBoost {ap_x:.3f}, Isolation Forest {ap_i:.3f}",
              f"Attack base rate in test set: {yte.mean():.4f}"]

    imp = pd.Series(xgb.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(8)
    lines += ["", "Top XGBoost features:", imp.to_markdown(floatfmt=".3f")]

    # Figure 1: precision-recall curves
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for scores, label in [(p_xgb, f"XGBoost, supervised (AP {ap_x:.2f})"),
                          (s_te, f"Isolation Forest, no attack labels (AP {ap_i:.2f})")]:
        pr, rc, _ = precision_recall_curve(yte, scores)
        ax.plot(rc, pr, label=label, linewidth=2)
    ax.axhline(yte.mean(), ls="--", c="gray", lw=1, label="Random guess")
    ax.set_xlabel("Recall (share of attack contracts caught)")
    ax.set_ylabel("Precision (share of alerts that are real attacks)")
    ax.set_title("Flagging DeFi attack contracts at deployment")
    ax.legend(loc="lower left", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig("pr_curve.png", dpi=200)

    out = "\n".join(lines)
    open("results.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lookahead_features.csv")
