import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(
        self,
        input_dim=768,      # embedding size
        hidden_dim=256,
        latent_dim=32
    ):
        super().__init__()

        # Encoder
        self.enc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.mu = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )

        # Decoder
        self.dec1 = nn.Linear(
            latent_dim,
            hidden_dim
        )

        self.out = nn.Linear(
            hidden_dim,
            input_dim
        )

    def encode(self, x):

        h = F.relu(
            self.enc1(x)
        )

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

        z = mu + eps * std

        return z

    def decode(self, z):

        h = F.relu(
            self.dec1(z)
        )

        return self.out(h)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        recon = self.decode(z)

        return recon, mu, logvar, z


def vae_loss(
    recon,
    x,
    mu,
    logvar,
    beta=1.0
):
    recon_loss = F.mse_loss(
        recon,
        x,
        reduction="mean"
    )

    kl_loss = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = (
        recon_loss
        + beta * kl_loss
    )

    return (
        total_loss,
        recon_loss,
        kl_loss
    )


# --------------------------------------------------
# Example
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(
    input_dim=768,
    hidden_dim=256,
    latent_dim=32
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

for step in range(1000):

    x = torch.randn(
        64,
        768
    ).to(device)

    recon, mu, logvar, z = model(x)

    loss, recon_loss, kl_loss = vae_loss(
        recon,
        x,
        mu,
        logvar,
        beta=1.0
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"step={step} "
            f"loss={loss.item():.4f} "
            f"recon={recon_loss.item():.4f} "
            f"kl={kl_loss.item():.4f}"
        )

# --------------------------------------------------
# Latent vectors for PCA / classifier
# --------------------------------------------------

with torch.no_grad():

    embeddings = torch.randn(
        1000,
        768
    ).to(device)

    mu, logvar = model.encode(
        embeddings
    )

    latent_vectors = mu.cpu().numpy()

# PCA(latent_vectors)
# Classifier(PCA_output)
