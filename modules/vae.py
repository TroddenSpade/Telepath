import torch
import torch.nn as nn


class CycleVAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu_1 = nn.Linear(128, latent_size)
        self.fc_logvar_1 = nn.Linear(128, latent_size)
        self.decoder_1 = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu_2 = nn.Linear(128, latent_size)
        self.fc_logvar_2 = nn.Linear(128, latent_size)
        self.decoder_2 = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_1, x_2):
        h_1 = self.encoder_1(x_1)
        mu_1, logvar_1 = self.fc_mu_1(h_1), self.fc_logvar_1(h_1)
        z_1 = self.reparameterize(mu_1, logvar_1)

        h_2 = self.encoder_2(x_2)
        mu_2, logvar_2 = self.fc_mu_2(h_2), self.fc_logvar_2(h_2)
        z_2 = self.reparameterize(mu_2, logvar_2)

        recon_1 = self.decoder_1(z_1)
        recon_2 = self.decoder_2(z_2)

        _2_from_1 = self.decoder_2(z_1)
        _1_from_2 = self.decoder_1(z_2)

        h_1_ = self.encoder_1(_1_from_2)
        mu_1_, logvar_1_ = self.fc_mu_1(h_1_), self.fc_logvar_1(h_1_)
        z_1_ = self.reparameterize(mu_1_, logvar_1_)

        h_2_ = self.encoder_2(_2_from_1)
        mu_2_, logvar_2_ = self.fc_mu_2(h_2_), self.fc_logvar_2(h_2_)
        z_2_ = self.reparameterize(mu_2_, logvar_2_)

        recon_1_from_2 = self.decoder_1(z_2_)
        recon_2_from_1 = self.decoder_2(z_1_)
        
        return recon_1, recon_2, recon_1_from_2, recon_2_from_1, mu_1, logvar_1, mu_2, logvar_2, mu_1_, logvar_1_, mu_2_, logvar_2_

