# FraudGuard

[![CI](https://github.com/SadraMajidi04/fraudguard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/fraudguard/actions)

Cost-aware, explainable **transaction fraud detection** in Python: synthetic data, leak-free
features, four models, money-based evaluation, explanations, drift monitoring and a real-time
scoring API. **223 automated tests.**

The goal is not "train a classifier". It is the part that decides whether a fraud system works in
production: extreme class imbalance, decisions judged in **money**, no time leakage, no
training/serving skew, and a reason for every blocked transaction.

## Results

Test set = the last 20% of the timeline, never used for any choice. Alert threshold chosen on the
validation period to minimise cost. Cost model: each alert costs $5, a missed fraud costs its
amount. 100,000 transactions, 1% fraud.

| Model                            | PR-AUC [95% CI]          | ROC-AUC | Recall @ 1% alerts | Recall | Precision | Cost       | Savings vs approve-all |
| -------------------------------- | ------------------------ | ------- | ------------------ | ------ | --------- | ---------- | ---------------------- |
| rules (baseline)                 | 0.557 [0.486, 0.618]     | 0.884   | 0.56               | 0.57   | 0.53      | $7,091     | 88.4%                  |
| logistic regression              | 0.724 [0.656, 0.769]     | 0.983   | 0.65               | 0.64   | 0.69      | $6,854     | 88.8%                  |
| autoencoder (NN)                 | 0.448 [0.377, 0.510]     | 0.909   | 0.45               | 0.59   | 0.26      | $12,629    | 79.3%                  |
| **gradient boosting (selected)** | **0.860 [0.805, 0.898]** | 0.985   | 0.82               | 0.83   | 0.73      | **$4,897** | **92.0%**              |

"Approve everything" would cost $61,148 on the same test period. Full numbers:
[docs/results/](docs/results/).

**Where it fails (important):** the selected model catches 97-100% of three fraud patterns but only
**8% of "stealth takeover"**, fraud that imitates the victim's normal behaviour. That is by design
(and realistic): a model cannot flag what looks identical to normal traffic. See the
[model card](docs/MODEL_CARD.md).

> **Honest caveat:** the data is synthetic (real data is confidential). Results prove the
> _pipeline_ is sound, not that the model would catch real-world fraud. Run it on a public real
> dataset before making any real-world claim.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

pytest                                # 223 tests
make train                            # generate data, train 4 models, save models/model.joblib
make serve                            # start the API on http://127.0.0.1:8000
make drift                            # PSI drift report against the training distribution
```

Without `make`: `python -m fraudguard.cli train --out-dir models` (generates data itself),
then `python -m fraudguard.cli serve --model models/model.joblib`.

### Score a transaction

```bash
curl -s localhost:8000/score -H 'Content-Type: application/json' -d '{
  "transaction_id": "demo-1", "timestamp": "2025-04-15T03:12:00",
  "customer_id": "C000010", "merchant_id": "M00500", "merchant_category": "electronics",
  "channel": "online", "country": "NG", "customer_home_country": "US",
  "device_id": "X99999999", "amount": 1450.0}'
```

```json
{
  "transaction_id": "demo-1",
  "score": 1.0,
  "threshold": 0.338,
  "decision": "review",
  "reasons": [
    {
      "feature": "category_code",
      "score_increase": 0.0853,
      "text": "merchant category: electronics"
    },
    {
      "feature": "amount",
      "score_increase": 0.0056,
      "text": "transaction amount: 1,450.00"
    },
    {
      "feature": "is_foreign",
      "score_increase": 0.0004,
      "text": "made outside the customer's home country"
    }
  ],
  "warnings": [],
  "model": "gradient_boosting"
}
```

Endpoints: `GET /health`, `GET /model-info`, `POST /score`. Errors are JSON with proper codes
(400 bad JSON, 409 duplicate id, 413 too large, 415 wrong content type, 422 invalid field).

Docker (not tested in the environment this was written in): `docker build -t fraudguard .`
then `docker run -p 8000:8000 fraudguard`.

## How it works

```
transactions ─► validation ─► features (past only) ─► time split ─► 4 models
                                                                      │
        API  ◄─ model bundle ◄─ cost-optimal threshold (validation) ◄─┘
         │                          test set used once, for the report
         └─► same build_features() online ─► score + reasons          drift: PSI vs training
```

| Module                             | Role                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `data/synthetic.py`                | transaction generator, 4 fraud patterns, exact fraud count, seeded                     |
| `data/validation.py`, `data/io.py` | schema checks; bad data fails loudly, before any model sees it                         |
| `features.py`                      | 20 leak-free features (customer history, 1h/24h velocity, new device/country, ...)     |
| `splits.py`                        | train < validation < test **in time**                                                  |
| `models.py`                        | rules, logistic regression, gradient boosting (grid chosen on validation), autoencoder |
| `evaluation.py`                    | PR-AUC, recall at alert budget, cost model, cost-optimal threshold, bootstrap CI       |
| `explain.py`                       | permutation importance + per-transaction reason codes                                  |
| `monitoring.py`                    | PSI drift report                                                                       |
| `pipeline.py`                      | the whole experiment + model bundle save/load                                          |
| `serving.py`                       | online scorer + HTTP API (standard library only)                                       |

## Engineering decisions worth discussing

Full reasoning in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md). In short:

- **Money, not accuracy.** 99% of traffic is legitimate; the threshold minimises expected cost.
- **No leakage, proven by test:** deleting all future rows never changes any earlier row's
  features; changing labels never changes features; random labels give chance-level models.
- **No training/serving skew, proven by test:** scoring 500 transactions one by one through the API
  logic gives the same scores as batch scoring (to 1e-9).
- **Test set touched once;** the winner is picked by validation cost.
- **Explanations without SHAP:** occlusion-based reason codes work with any model and add no
  dependency. Trade-off: single-feature occlusion understates effects when a score is saturated.

## Testing

`pytest` runs 223 tests: generator realism per fraud pattern, 30+ malformed-data cases,
hand-computed feature values, the leakage/parity/label-shuffle tests above, a brute-force check of
the cost-optimal threshold on random data, PSI edge cases (constants, NaN, unseen values), a live
HTTP server with malformed requests, concurrency, and CLI end-to-end runs.

## Limitations and next steps

- Synthetic data only. Next: adapter for a public dataset (e.g. IEEE-CIS) and re-run.
- The "neural network" is a small autoencoder built with scikit-learn's MLP. A PyTorch
  version, sequence models over a customer's transaction history, and graph models (shared
  devices/merchants) are the natural deep-learning extensions.
- The API keeps customer history in memory (a stand-in for a feature store such as Redis); history
  per customer is capped (default 10,000), and it is lost on restart unless warmed with `--warm-data`.
- No authentication/TLS on the API: put it behind a gateway. Model files use pickle (joblib):
  only load files you trained yourself.
- Labels are assumed immediate; real chargeback labels arrive weeks late.
