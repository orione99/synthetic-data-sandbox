import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TabularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # Output in [0, 1] coerente con MinMaxScaler e OneHot
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class VAETrainer:
    def __init__(self, input_dim: int, latent_dim: int = 8, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TabularVAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction Loss (MSE media per campione)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def train(
        self,
        data: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        max_beta: float = 0.1,
    ) -> list[dict]:
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        # drop_last=True evita batch da 1 elemento che romperebbero BatchNorm1d
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(data) > batch_size)

        history = []
        self.model.train()

        for epoch in range(1, epochs + 1):
            # KL Annealing lineare nella prima metà delle epoche
            beta = min(max_beta, max_beta * (epoch / max(1, epochs // 2)))
            epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()

                recon_x, mu, logvar = self.model(batch_x)
                loss, recon_loss, kl_loss = self.loss_function(recon_x, batch_x, mu, logvar, beta=beta)

                loss.backward()
                self.optimizer.step()

                epoch_total += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            n_batches = len(loader)
            history.append({
                "epoch": epoch,
                "total_loss": round(epoch_total / n_batches, 5),
                "recon_loss": round(epoch_recon / n_batches, 5),
                "kl_loss": round(epoch_kl / n_batches, 5),
            })

        return history

    @torch.no_grad()
    def generate(self, num_samples: int) -> np.ndarray:
        self.model.eval()
        z = torch.randn(num_samples, self.model.latent_dim, device=self.device)
        synthetic_tensor = self.model.decode(z)
        return synthetic_tensor.cpu().numpy()


# Test rapido eseguibile direttamente da terminale
if __name__ == "__main__":
    import pandas as pd
    from data_engine import TabularDataEngine

    df_real = pd.read_csv("data/diabetes.csv")
    print(f"Dataset reale caricato: {df_real.shape}")

    engine = TabularDataEngine()
    X_scaled = engine.fit_transform(df_real)

    trainer = VAETrainer(input_dim=X_scaled.shape[1], latent_dim=8, lr=1e-3)
    print(f"Training su device: {trainer.device}")
    history = trainer.train(X_scaled, epochs=150, batch_size=32)
    print("Ultima epoca:", history[-1])

    raw_synth = trainer.generate(num_samples=len(df_real))
    df_synth = engine.inverse_transform(raw_synth)

    print("\nPrime 5 righe sintetiche generate:")
    print(df_synth.head())
    print("\nConfronto medie (Reale vs Sintetico):")
    comparison = pd.DataFrame({"Reale": df_real.mean(), "Sintetico": df_synth.mean()})
    print(comparison.round(2))