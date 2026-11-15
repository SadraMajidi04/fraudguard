# Problem definition

## Business problem
Card-not-present and account-takeover fraud costs payment companies and banks billions
per year. A fraud system must decide, in milliseconds, for every transaction:
**approve, challenge (step-up auth / manual review), or block.**

Two kinds of mistakes, with very different prices:

| Mistake | What happens | Cost |
|---|---|---|
| False negative (missed fraud) | money is lost + chargeback fees | ~ the transaction amount |
| False positive (good customer blocked) | review work + customer friction / churn | small fixed cost per alert |

Therefore accuracy is a meaningless metric (99% of transactions are legitimate, so
"approve everything" is 99% accurate and useless). This project optimizes **expected
monetary cost**, not accuracy.

## Evaluation (all implemented in this repository)
* **PR-AUC** (not ROC-AUC) as the threshold-free metric, because of class imbalance.
* **Recall at a fixed alert budget** (e.g. review at most 0.5% of traffic), which is how
  fraud teams actually operate.
* **Expected cost** with the table above; the decision threshold is chosen to minimize it.
* **Time-based split** (train on the past, test on the future). Random splits leak
  future information and overstate performance.
* **Explainability:** every flagged transaction gets human-readable reason codes
  (occlusion-based, works with any model) plus global permutation importance.
* **Drift monitoring**: fraud patterns change; we measure feature drift (PSI).

## Data
This project uses a **synthetic generator** with three documented fraud patterns
(`account_takeover`, `card_testing`, `atm_cashout`). Reasons: real data is confidential,
and a known ground truth lets us test the pipeline rigorously.

**Honest limitation:** a model can only learn patterns that exist in the data. Good
results on synthetic data prove the *pipeline* works, not that the model would catch real
fraud. The natural next step is an adapter for a public real dataset (e.g. IEEE-CIS Fraud)
that maps it to the schema below, then re-running the same pipeline.

## Canonical schema
`transaction_id, timestamp, customer_id, merchant_id, merchant_category, channel, country,
customer_home_country, device_id, amount, is_fraud`

`fraud_pattern` exists only for analysis (`include_pattern=True`) and must never be used
as a feature: it is the label in disguise (target leakage).
