import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from build import preprocess, chrono_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from xgboost import XGBClassifier
df = pd.read_csv("lookahead_features.csv", dtype=str); df["tag"]=df.tag.astype(int); df["order_key"]=df.order_key.astype(float)
def run(d, label, seeds=(0,1,2,3,4)):
    tr, te = chrono_split(d); Xtr, Xte = preprocess(tr), preprocess(te).reindex(columns=preprocess(tr).columns, fill_value=0)
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte); ytr, yte = tr.tag.values, te.tag.values
    r = []
    for s in seeds:
        m = XGBClassifier(eval_metric="logloss", random_state=s).fit(A, ytr); p = m.predict_proba(B)[:,1]
        i = IsolationForest(n_estimators=300, random_state=s).fit(A[ytr==0]); q = -i.score_samples(B)
        thr = np.percentile(-i.score_samples(A[ytr==0]), 99)
        r.append([precision_score(yte,p>=.5), recall_score(yte,p>=.5), average_precision_score(yte,p), roc_auc_score(yte,p),
                  precision_score(yte,q>=thr,zero_division=0), recall_score(yte,q>=thr), average_precision_score(yte,q), roc_auc_score(yte,q)])
    r = np.array(r); names=["XGB prec","XGB rec","XGB AP","XGB AUC","IF prec","IF rec","IF AP","IF AUC"]
    print(label, f"(test attacks={yte.sum()}, test benign={len(yte)-yte.sum()})")
    for n, m, s in zip(names, r.mean(0), r.std(0)): print(f"  {n:9s} {m:.3f} ± {s:.3f}")
run(df, "ALL")
run(df[(df.tag==0)|(df.chain=="eth")], "ETH-ONLY attacks")
# what fraction of attacks have flash-loan callbacks vs benign
for t in (0,1):
    x = df[df.tag==t]; print(t, "flashloanCallbackNum>0:", (x.flashloanCallbackNum.astype(float)>0).mean().round(3), "verified:", (x.verify_tag=="True").mean().round(3), "Anonymous funding:", (x.fund_from_label==" Anonymous").mean().round(3))
