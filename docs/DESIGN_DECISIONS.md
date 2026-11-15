# Design decisions (and the questions an interviewer will ask)

**Why not accuracy or ROC-AUC?** With 1% fraud, "approve everything" is 99% accurate. ROC-AUC
looks great even for weak models on imbalanced data (0.88 for hand-written rules here). PR-AUC
reflects precision at the top of the ranking, and the real decision is monetary, so the threshold
is chosen to minimise `review_cost * alerts + missed_fraud_amount`.

**Why a time-based split?** A random split lets the model see the future (same customer, same
campaign, neighbouring days) and overstates performance. Train < validation < test in time mirrors
deployment. Features are computed on the full timeline but each row only sees its own past, so this
is safe (tested).

**How do you know there is no leakage?** Three tests: (1) truncating the data never changes earlier
rows' features, (2) flipping labels never changes features, (3) with shuffled labels every model
falls to chance. The `fraud_pattern` column exists only for analysis and is never a feature.

**Why choose the threshold on validation, then report on test?** Choosing it on test would report a
number that was tuned on the answer sheet. The winning model is also chosen by validation cost.

**What is training/serving skew and how is it prevented?** Different code paths for batch and live
features silently disagree. Here the API calls the same `build_features` on the customer's history.
A test replays 500 transactions one by one and compares with batch scores.

**Why gradient boosting wins here?** Tabular data with interactions (amount x channel x country x
history). Logistic regression is a solid baseline; the autoencoder is unsupervised (trained on
legitimate traffic only) and is useful when labels are scarce, but here it is weaker because
supervised labels are available.

**Why no SHAP?** Not installable in the build environment, and unnecessary: occlusion (replace a
feature by a typical value and measure the score drop) works with any model. Limitation: for
saturated scores or strongly correlated features, effects are understated. SHAP is a natural upgrade.

**Why a stdlib HTTP server instead of FastAPI?** Zero extra dependencies and fully testable here.
FastAPI would add validation docs and async support; `OnlineScorer` is framework-independent, so
swapping the transport layer is a small change.

**What would you change with more time?** Real data and delayed labels; a feature store; refit on
train+validation before release; calibrate probabilities; PyTorch sequence/graph models; fairness
audit of the `is_foreign` feature (it can penalise travellers and migrants); authentication and
rate limiting on the API.
