# Model card: FraudGuard transaction scorer

**Intended use:** ranking card transactions by fraud risk so that analysts review the riskiest
ones. A human review/step-up step is assumed; the model does not block customers by itself.
**Not for:** automatic account closure, credit decisions, or any use on real customers before
validation on real data.

## Data
Synthetic: 100,000 transactions, 1% fraud, 2,000 customers, 90 days, generated with
`fraudguard.data.generate_transactions(seed=42)`. Four fraud patterns: account takeover, card
testing (bursts of tiny purchases), ATM cash-out, and stealth takeover (mimics normal behaviour).
Train/validation/test = first 60% / next 20% / last 20% of the timeline.

## Model
Selected by validation cost: gradient boosting (scikit-learn `HistGradientBoostingClassifier`,
class-weighted, 20 features built from the transaction and the customer's earlier history).
Deployed threshold (0.338) minimises cost on the validation period ($5 per alert, missed fraud costs
its amount).

## Performance (test period, 20,000 transactions, 209 frauds)
PR-AUC 0.860 (95% CI 0.805-0.898), ROC-AUC 0.985, recall 0.83, precision 0.73. Cost $4,897 vs
$61,148 for approving everything (92.0% lower). Per fraud pattern at the deployed threshold:

| Pattern | Frauds | Caught | Recall |
|---|---|---|---|
| ATM cash-out | 46 | 46 | 100% |
| Card testing | 51 | 51 | 100% |
| Account takeover | 76 | 74 | 97% |
| **Stealth takeover** | 36 | 3 | **8%** |

Reproduce: `recall_by_pattern(df, bundle["model"], bundle["threshold"])` in `pipeline.py`
(generate with `include_pattern=True`). Most important features (permutation importance on
PR-AUC): amount, is_online, merchant category, is_foreign, time since previous transaction,
amount relative to the customer's usual amount.

## Limitations and risks
* Synthetic data: the card-testing result partly reflects that velocity features were designed
  with that pattern in mind. Real-world performance will be lower.
* Stealth fraud (looks like the victim) is largely invisible; ~$3.4k of missed amount on the test
  period comes from it. Needs extra signals (device binding, behavioural biometrics, user reports).
* Fairness: `is_foreign` and new-country flags may burden travellers and migrants; audit false-positive
  rates by group before any real deployment.
* Cost assumptions ($5 review, 100% loss) are placeholders; change them with
  `--review-cost` or `CostModel`.
* Concept drift: fraud patterns change. Run `fraudguard drift` regularly (exit code 3 means a
  major shift, PSI >= 0.25) and retrain.
* Probabilities are not calibrated; use the score only for ranking and the validated threshold.
