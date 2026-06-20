import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModel
)

from datasets import load_dataset

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.model_selection import train_test_split


# =====================================================
# CONFIG
# =====================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_NAME = "bert-base-uncased"

MAX_LEN = 128

LATENT_DIM = 32

HIDDEN_DIM = 256

BATCH_SIZE = 128

VAE_EPOCHS = 20

LR = 1e-3

NUM_LABELED = 1000

PCA_COMPONENTS = 16


# =====================================================
# LOAD DATA
# =====================================================

dataset = load_dataset(
    "ag_news"
)

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]


# =====================================================
# BERT
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert = AutoModel.from_pretrained(
    MODEL_NAME
)

bert.to(DEVICE)

bert.eval()

for p in bert.parameters():
    p.requires_grad = False


# =====================================================
# EMBEDDING FUNCTION
# =====================================================

@torch.no_grad()
def build_embeddings(
    texts,
    batch_size=64
):

    vectors = []

    for i in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            i:i+batch_size
        ]

        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        tok = {
            k:v.to(DEVICE)
            for k,v in tok.items()
        }

        out = bert(
            **tok
        )

        emb = out.last_hidden_state[
            :,0,:      # CLS token
        ]

        vectors.append(
            emb.cpu()
        )

    return torch.cat(vectors)


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

print("Building train embeddings")

train_embeddings = build_embeddings(
    train_texts
)

print("Building test embeddings")

test_embeddings = build_embeddings(
    test_texts
)

X_train_full = train_embeddings.numpy()

X_test_full = test_embeddings.numpy()

y_train_full = np.array(
    train_labels
)

y_test = np.array(
    test_labels
)


# =====================================================
# SMALL LABEL SIMULATION
# =====================================================

idx = np.random.choice(
    len(X_train_full),
    NUM_LABELED,
    replace=False
)

X_labeled = X_train_full[idx]

y_labeled = y_train_full[idx]


# =====================================================
# VAE
# =====================================================

class VAE(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        latent_dim=32
    ):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
        )

        self.mu = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.dec = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def encode(
        self,
        x
    ):

        h = self.enc(x)

        mu = self.mu(h)

        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return mu + eps * std

    def decode(
        self,
        z
    ):

        return self.dec(z)

    def forward(
        self,
        x
    ):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        recon = self.decode(z)

        return (
            recon,
            mu,
            logvar,
            z
        )


def vae_loss(
    recon,
    x,
    mu,
    logvar
):

    recon_loss = F.mse_loss(
        recon,
        x
    )

    kl = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return recon_loss + kl


# =====================================================
# TRAIN VAE ON ALL TEXTS
# =====================================================

vae = VAE(
    latent_dim=LATENT_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    vae.parameters(),
    lr=LR
)

tensor_data = TensorDataset(
    torch.tensor(
        X_train_full,
        dtype=torch.float32
    )
)

loader = DataLoader(
    tensor_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(
    VAE_EPOCHS
):

    vae.train()

    losses = []

    for (x,) in loader:

        x = x.to(DEVICE)

        recon, mu, logvar, z = vae(x)

        loss = vae_loss(
            recon,
            x,
            mu,
            logvar
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(
            loss.item()
        )

    print(
        epoch,
        np.mean(losses)
    )


# =====================================================
# LATENT FEATURES
# =====================================================

@torch.no_grad()
def get_latent(
    X
):

    X = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    mu, logvar = vae.encode(
        X
    )

    return mu.cpu().numpy()


latent_train_full = get_latent(
    X_train_full
)

latent_test = get_latent(
    X_test_full
)

latent_labeled = latent_train_full[
    idx
]


# =====================================================
# PCA
# =====================================================

pca = PCA(
    n_components=PCA_COMPONENTS
)

pca.fit(
    latent_train_full
)

train_pca = pca.transform(
    latent_labeled
)

test_pca = pca.transform(
    latent_test
)


# =====================================================
# CLASSIFIER
# =====================================================

clf = LogisticRegression(
    max_iter=5000
)

clf.fit(
    train_pca,
    y_labeled
)

pred = clf.predict(
    test_pca
)


# =====================================================
# RESULTS
# =====================================================

acc = accuracy_score(
    y_test,
    pred
)

print()
print(
    f"Accuracy = {acc:.4f}"
)

print()

print(
    classification_report(
        y_test,
        pred
    )
)
