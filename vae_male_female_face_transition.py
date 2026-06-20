# ============================================================
# VAE Gender Knob Demo
# ------------------------------------------------------------
# Downloads CelebA
# Trains a Variational Autoencoder
# Learns a latent "Male ↔ Female" direction
# Generates smooth interpolation images
#
# Output:
#   reconstructions.png
#   latent_space_pca.png
#   gender_knob.png
#
# GPU recommended
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.utils import make_grid

from sklearn.decomposition import PCA


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64

LATENT_DIM = 128

BATCH_SIZE = 128

EPOCHS = 20

LR = 1e-4

BETA = 1.0

NUM_INTERPOLATION_STEPS = 11


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose(
    [
        transforms.CenterCrop(178),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

dataset = CelebA(
    root="./data",
    split="train",
    target_type="attr",
    download=True,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(
            256 * 4 * 4,
            LATENT_DIM
        )

        self.fc_logvar = nn.Linear(
            256 * 4 * 4,
            LATENT_DIM
        )

        self.fc_decode = nn.Linear(
            LATENT_DIM,
            256 * 4 * 4
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                4,
                2,
                1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(
            x.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(
        self,
        mu,
        logvar
    ):
        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(
        self,
        z
    ):
        h = self.fc_decode(z)

        h = h.view(
            -1,
            256,
            4,
            4
        )

        return self.decoder(h)

    def forward(self, x):

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


# ============================================================
# LOSS
# ============================================================

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

    total = recon_loss + BETA * kl

    return total, recon_loss, kl


# ============================================================
# TRAIN
# ============================================================

model = VAE().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for imgs, attrs in loader:

        imgs = imgs.to(DEVICE)

        recon, mu, logvar, z = model(imgs)

        loss, rloss, kl = vae_loss(
            recon,
            imgs,
            mu,
            logvar
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={running_loss/len(loader):.4f}"
    )


# ============================================================
# RECONSTRUCTION EXAMPLE
# ============================================================

model.eval()

imgs, attrs = next(iter(loader))

imgs = imgs[:8].to(DEVICE)

with torch.no_grad():

    recon, mu, logvar, z = model(imgs)

comparison = torch.cat(
    [imgs.cpu(), recon.cpu()]
)

grid = make_grid(
    comparison,
    nrow=8
)

plt.figure(figsize=(16, 4))
plt.imshow(
    grid.permute(1, 2, 0)
)
plt.axis("off")
plt.savefig(
    "reconstructions.png",
    bbox_inches="tight"
)
plt.close()


# ============================================================
# LATENT COLLECTION
# ============================================================

all_z = []

all_gender = []

MALE_INDEX = 20

with torch.no_grad():

    for imgs, attrs in loader:

        imgs = imgs.to(DEVICE)

        mu, logvar = model.encode(
            imgs
        )

        all_z.append(
            mu.cpu()
        )

        gender = attrs[:, MALE_INDEX]

        all_gender.append(
            gender
        )

        if len(all_z) > 100:
            break

all_z = torch.cat(all_z).numpy()

all_gender = torch.cat(
    all_gender
).numpy()


# ============================================================
# PCA VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)

z2 = pca.fit_transform(
    all_z
)

plt.figure(figsize=(8, 6))

plt.scatter(
    z2[:, 0],
    z2[:, 1],
    c=all_gender,
    s=8,
    alpha=0.5
)

plt.title(
    "Latent Space PCA"
)

plt.savefig(
    "latent_space_pca.png"
)

plt.close()


# ============================================================
# GENDER DIRECTION
# ============================================================

male_mask = all_gender == 1

female_mask = all_gender == -1

male_mean = all_z[
    male_mask
].mean(axis=0)

female_mean = all_z[
    female_mask
].mean(axis=0)

gender_direction = (
    female_mean
    - male_mean
)

gender_direction = (
    gender_direction
    / np.linalg.norm(
        gender_direction
    )
)


# ============================================================
# PICK MALE FACE
# ============================================================

for imgs, attrs in loader:

    male_idx = torch.where(
        attrs[:, MALE_INDEX] == 1
    )[0]

    if len(male_idx):

        face = imgs[
            male_idx[0]
        ].unsqueeze(0)

        break

face = face.to(DEVICE)

with torch.no_grad():

    mu, logvar = model.encode(
        face
    )

base_z = (
    mu.squeeze()
    .cpu()
    .numpy()
)


# ============================================================
# SMOOTH MAN ↔ WOMAN KNOB
# ============================================================

generated = []

for alpha in np.linspace(
    -4,
    4,
    NUM_INTERPOLATION_STEPS
):

    z_new = (
        base_z
        + alpha
        * gender_direction
    )

    z_tensor = torch.tensor(
        z_new,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        img = model.decode(
            z_tensor
        )

    generated.append(
        img.cpu()
    )

generated = torch.cat(
    generated
)

grid = make_grid(
    generated,
    nrow=NUM_INTERPOLATION_STEPS
)

plt.figure(
    figsize=(24, 4)
)

plt.imshow(
    grid.permute(1, 2, 0)
)

plt.axis("off")

plt.title(
    "Male ↔ Female Latent Knob"
)

plt.savefig(
    "gender_knob.png",
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print("  reconstructions.png")
print("  latent_space_pca.png")
print("  gender_knob.png")
