import torch
import torch.nn as nn

# ------------------------------
# CLSTM Decoder
# ------------------------------
class CLSTM_Decoder(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=4):
        super(CLSTM_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # LSTM transformation layer.
        # We pass the concatenated latent vector and condition (shape: latent_dim+condition_dim) as a sequence of length 1.
        self.lstm = nn.LSTM(input_size=latent_dim + condition_dim, hidden_size=latent_dim,
                            num_layers=1, batch_first=True)
        
        # Decoder layers: Produce output of shape (batch, 4, 200, 128)
        # We'll start by mapping the transformed latent+condition vector to a tensor that can be reshaped.
        # We choose an initial shape of (128, 25, 16) so that after three layers of upsampling we reach (200, 128):
        # Height: 25 -> 50 -> 100 -> 200; Width: 16 -> 32 -> 64 -> 128.
        self.fc_decode = nn.Linear(latent_dim + condition_dim, 128 * 25 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_relu = nn.ReLU()

    def transform_latent(self, z, condition):
        # Concatenate latent vector and condition.
        z_cat = torch.cat([z, condition], dim=1)  # (batch, latent_dim + condition_dim)
        # Unsqueeze to create a sequence of length 1.
        z_cat_seq = z_cat.unsqueeze(1)  # (batch, 1, latent_dim + condition_dim)
        lstm_out, _ = self.lstm(z_cat_seq)  # lstm_out: (batch, 1, latent_dim)
        z_transformed = lstm_out.squeeze(1)  # (batch, latent_dim)
        return z_transformed

    def decode(self, z_transformed, condition):
        # Concatenate the transformed latent vector with the condition.
        dec_input = torch.cat([z_transformed, condition], dim=1)  # (batch, latent_dim + condition_dim)
        h = self.fc_decode(dec_input)
        h = h.view(-1, 128, 25, 16)  # Reshape to (batch, 128, 25, 16)
        h = self.dec_relu(self.dec_conv1(h))  # (batch, 64, 50, 32)
        h = self.dec_relu(self.dec_conv2(h))  # (batch, 32, 100, 64)
        logits = self.dec_conv3(h)            # (batch, 4, 200, 128)
        return logits

    def forward(self, z, condition):
        z_transformed = self.transform_latent(z, condition)
        logits = self.decode(z_transformed, condition)
        return logits