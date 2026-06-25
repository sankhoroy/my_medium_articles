# bayesian_house_price_pyro.py

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

# ============================================================
# 1. CREATE SYNTHETIC HOUSE PRICE DATA
# ============================================================

np.random.seed(42)
torch.manual_seed(42)
pyro.set_rng_seed(42)

N = 1000

area = np.random.normal(2000, 500, N)
bedrooms = np.random.randint(1, 6, N)
age = np.random.randint(0, 40, N)

price = (
    120 * area
    + 25000 * bedrooms
    - 1500 * age
    + np.random.normal(0, 30000, N)
)

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "price": price
})

print("\nSample Data")
print(df.head())

# ============================================================
# 2. PREPARE DATA
# ============================================================

X = df[["area", "bedrooms", "age"]].values
y = df["price"].values

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)

y_mean = y.mean()
y_std = y.std()

y_scaled = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_scaled,
    test_size=0.20,
    random_state=42
)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# ============================================================
# 3. BAYESIAN NEURAL NETWORK
# ============================================================

def model(x, y=None):

    input_dim = x.shape[1]
    hidden_dim = 16

    # Layer 1 weights
    w1 = pyro.sample(
        "w1",
        dist.Normal(
            torch.zeros(input_dim, hidden_dim),
            torch.ones(input_dim, hidden_dim)
        ).to_event(2)
    )

    b1 = pyro.sample(
        "b1",
        dist.Normal(
            torch.zeros(hidden_dim),
            torch.ones(hidden_dim)
        ).to_event(1)
    )

    # Layer 2 weights
    w2 = pyro.sample(
        "w2",
        dist.Normal(
            torch.zeros(hidden_dim, 1),
            torch.ones(hidden_dim, 1)
        ).to_event(2)
    )

    b2 = pyro.sample(
        "b2",
        dist.Normal(
            torch.zeros(1),
            torch.ones(1)
        ).to_event(1)
    )

    hidden = torch.relu(x @ w1 + b1)

    mean = hidden @ w2 + b2
    mean = mean.squeeze(-1)

    sigma = pyro.sample(
        "sigma",
        dist.Uniform(0.01, 1.0)
    )

    with pyro.plate("data", x.shape[0]):
        pyro.sample(
            "obs",
            dist.Normal(mean, sigma),
            obs=y
        )

# ============================================================
# 4. GUIDE (APPROXIMATE POSTERIOR)
# ============================================================

guide = AutoDiagonalNormal(model)

# ============================================================
# 5. TRAINING
# ============================================================

pyro.clear_param_store()

optimizer = Adam({"lr": 0.01})

svi = SVI(
    model,
    guide,
    optimizer,
    loss=Trace_ELBO()
)

losses = []

print("\nTraining Bayesian Neural Network...\n")

for step in range(3000):

    loss = svi.step(
        X_train,
        y_train
    )

    losses.append(loss)

    if step % 200 == 0:
        print(f"Step {step:4d}  Loss = {loss:.2f}")

# ============================================================
# 6. TRAINING CURVE
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("ELBO Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ============================================================
# 7. TEST PREDICTIONS
# ============================================================

predictive = Predictive(
    model,
    guide=guide,
    num_samples=1000
)

samples = predictive(X_test)

predictions = samples["obs"]

mean_pred = predictions.mean(0)
lower_pred = predictions.quantile(0.025, dim=0)
upper_pred = predictions.quantile(0.975, dim=0)

# back to original dollar scale

mean_pred = mean_pred.numpy() * y_std + y_mean
lower_pred = lower_pred.numpy() * y_std + y_mean
upper_pred = upper_pred.numpy() * y_std + y_mean

actual = y_test.numpy() * y_std + y_mean

# ============================================================
# 8. VISUALIZE TEST PREDICTIONS
# ============================================================

idx = np.arange(len(actual))

plt.figure(figsize=(12,6))

plt.scatter(
    idx,
    actual,
    label="Actual Price",
    alpha=0.7
)

plt.plot(
    idx,
    mean_pred,
    linewidth=2,
    label="Mean Prediction"
)

plt.fill_between(
    idx,
    lower_pred,
    upper_pred,
    alpha=0.25,
    label="95% Confidence Interval"
)

plt.xlabel("Test House")
plt.ylabel("Price")
plt.title("Bayesian Neural Network Predictions")
plt.legend()
plt.show()

# ============================================================
# 9. PREDICT A NEW HOUSE
# ============================================================

new_house = np.array([
    [2200, 3, 10]
])

scaled_house = x_scaler.transform(new_house)
scaled_house = torch.tensor(scaled_house).float()

predictive = Predictive(
    model,
    guide=guide,
    num_samples=2000
)

samples = predictive(scaled_house)

preds = samples["obs"]

mean_price = preds.mean().item()
low_price = preds.quantile(0.025).item()
high_price = preds.quantile(0.975).item()

mean_price = mean_price * y_std + y_mean
low_price = low_price * y_std + y_mean
high_price = high_price * y_std + y_mean

print("\n")
print("=" * 60)
print("HOUSE PRICE PREDICTION")
print("=" * 60)

print(f"Area      : 2200 sq ft")
print(f"Bedrooms  : 3")
print(f"Age       : 10 years")

print("\nBayesian Prediction")

print(f"Mean Price      : ${mean_price:,.0f}")
print(f"95% Lower Bound : ${low_price:,.0f}")
print(f"95% Upper Bound : ${high_price:,.0f}")

print("=" * 60)

# ============================================================
# 10. VISUALIZE PREDICTIVE DISTRIBUTION
# ============================================================

preds = preds.numpy()
preds = preds * y_std + y_mean

plt.figure(figsize=(8,5))

plt.hist(
    preds,
    bins=40
)

plt.axvline(mean_price)

plt.title(
    "Posterior Predictive Distribution\n"
    "House: 2200 sqft, 3 bed, 10 years old"
)

plt.xlabel("Predicted Price")
plt.ylabel("Frequency")

plt.show()
