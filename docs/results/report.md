Rows: 100,000 (1,000 fraud, 1.00%); train/valid/test = 60,000/20,000/20,000 (by time).
Cost model: alert = $5, missed fraud = 1 x amount. Alert budget for recall: 1.00% of traffic.

| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ budget | Recall | Precision | Cost | Savings vs approve-all |
|---|---|---|---|---|---|---|---|
| rules | 0.557 [0.486, 0.618] | 0.884 | 0.56 | 0.57 | 0.53 | $7,091 | 88.4% |
| logistic_regression | 0.724 [0.656, 0.769] | 0.983 | 0.65 | 0.64 | 0.69 | $6,854 | 88.8% |
| autoencoder | 0.448 [0.377, 0.510] | 0.909 | 0.45 | 0.59 | 0.26 | $12,629 | 79.3% |
| gradient_boosting **(selected)** | 0.860 [0.805, 0.898] | 0.985 | 0.82 | 0.83 | 0.73 | $4,897 | 92.0% |

Approve-everything baseline cost on test: $61,148.
