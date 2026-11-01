train: 12876 contracts (300 attacks) | test: 3220 contracts (75 attacks)

| model                             |   precision |   recall |    f1 |   FPR % |   TP |   FN |   FP |   TN |
|:----------------------------------|------------:|---------:|------:|--------:|-----:|-----:|-----:|-----:|
| XGBoost (supervised, replication) |       0.771 |    0.853 | 0.810 |   0.604 |   64 |   11 |   19 | 3126 |
| Isolation Forest, threshold p99.0 |       0.341 |    0.187 | 0.241 |   0.859 |   14 |   61 |   27 | 3118 |
| Isolation Forest, threshold p99.5 |       0.211 |    0.053 | 0.085 |   0.477 |    4 |   71 |   15 | 3130 |

Average precision (area under PR curve): XGBoost 0.901, Isolation Forest 0.208
Attack base rate in test set: 0.0233

Top XGBoost features:
|                       |     0 |
|:----------------------|------:|
| flashloanCallbackProp | 0.418 |
| fund_Safe             | 0.191 |
| token0CallCount       | 0.093 |
| verify_tag            | 0.045 |
| avgTokenCallNum       | 0.037 |
| transferFromCallCount | 0.034 |
| fund_unknown_high     | 0.021 |
| fund_ Anonymous       | 0.018 |