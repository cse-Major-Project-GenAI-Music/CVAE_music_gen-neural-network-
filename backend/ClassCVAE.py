import torch
import torch.nn as nn
# ------------------------------
# CVAE Model Class
# ------------------------------
class CVAE(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=4):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        # Encoder layers remain the same.
        self.enc_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_relu = nn.ReLU()
        self.flatten_dim = 128 * 38 * 16  # unchanged

        # Now fc_mu and fc_logvar expect an input with added condition dimension.
        self.fc_mu = nn.Linear(self.flatten_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + condition_dim, latent_dim)

        # Decoder: now latent vector is concatenated with condition.
        self.fc_decode = nn.Linear(latent_dim + condition_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0,1))
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1))
        self.dec_conv3 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=(1,1))
        self.dec_relu = nn.ReLU()

    def encode(self, x, condition):
        h = self.enc_relu(self.enc_conv1(x))
        h = self.enc_relu(self.enc_conv2(h))
        h = self.enc_relu(self.enc_conv3(h))
        h = h.view(-1, self.flatten_dim)
        # Concatenate condition to flattened features.
        h = torch.cat([h, condition], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        # Concatenate condition to latent vector.
        z = torch.cat([z, condition], dim=1)
        h = self.fc_decode(z)
        h = h.view(-1, 128, 38, 16)
        h = self.dec_relu(self.dec_conv1(h))
        h = self.dec_relu(self.dec_conv2(h))
        logits = self.dec_conv3(h)
        return logits

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, condition)
        return logits, mu, logvar