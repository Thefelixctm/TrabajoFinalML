import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_size) # La capa del "cuello de botella"
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            nn.Sigmoid() # Sigmoid para que la salida esté entre 0 y 1, como las imágenes normalizadas
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x