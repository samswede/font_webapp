
import torch
from torch import nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        ### Convolutions 1
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        ### Convolutions 2
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.linear1 = nn.Linear(256*32*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)
        
        ### Probabilistic section
        self.N = torch.distributions.Normal(0, 1)

        self.kl = 0

    def forward(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        log_sigma = self.linear3(x)
        sigma = torch.exp(log_sigma)

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - sigma.pow(2))).mean()

        return z
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256*32*32),
            nn.ReLU()
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 32, 32))

        ### Convolutional section 1
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        ### Convolutional section 2
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = torch.sigmoid(x)
        return x
    

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
