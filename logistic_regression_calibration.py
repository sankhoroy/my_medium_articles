from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss
)

# ----------------------------------------------------
# Create sample dataset
# ----------------------------------------------------
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    random_state=42
)

# ----------------------------------------------------
# Train / Validation / Test Split
# ----------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# ----------------------------------------------------
# Train Logistic Regression
# ----------------------------------------------------
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# ----------------------------------------------------
# Standard Metrics
# ----------------------------------------------------
pred_test = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred_test))

print("F1:",
      f1_score(y_test, pred_test))

# Raw probabilities
prob_test = model.predict_proba(X_test)[:, 1]

print(
    "Brier Score (Before Calibration):",
    brier_score_loss(y_test, prob_test)
)
```

At this point, most machine learning projects stop.

However, we still don't know whether a predicted probability of 0.80 truly corresponds to an 80% chance of the event occurring.

Let's calibrate.

### Isotonic Regression Calibration

```python
calibrated_model = CalibratedClassifierCV(
    estimator=model,
    method="isotonic",
    cv="prefit"
)

calibrated_model.fit(
    X_valid,
    y_valid
)

calibrated_probs = calibrated_model.predict_proba(
    X_test
)[:, 1]

print(
    "Brier Score (After Calibration):",
    brier_score_loss(
        y_test,
        calibrated_probs
    )
)
```

If calibration is successful, the Brier Score should decrease, indicating that predicted probabilities are now closer to real-world frequencies.

### Platt Scaling Calibration

```python
platt_model = CalibratedClassifierCV(
    estimator=model,
    method="sigmoid",
    cv="prefit"
)

platt_model.fit(
    X_valid,
    y_valid
)

platt_probs = platt_model.predict_proba(
    X_test
)[:, 1]

print(
    "Platt Brier Score:",
    brier_score_loss(
        y_test,
        platt_probs
    )
)
```

The best calibration method is usually selected based on calibration metrics such as:

* Brier Score
* Expected Calibration Error (ECE)
* Reliability Diagram

rather than Accuracy or F1.

### Business Threshold Optimization

Once probabilities are calibrated, threshold selection becomes meaningful.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

TP_VALUE = 500
FP_COST = 10
FN_COST = 1000

best_profit = -1e18
best_threshold = None

for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    preds = (
        calibrated_probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        preds
    ).ravel()

    profit = (
        tp * TP_VALUE
        - fp * FP_COST
        - fn * FN_COST
    )

    if profit > best_profit:
        best_profit = profit
        best_threshold = threshold

print(
    f"Best Threshold: {best_threshold:.2f}"
)

print(
    f"Maximum Profit: ${best_profit:,.0f}"
)
