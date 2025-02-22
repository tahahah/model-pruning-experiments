from dcaecore.pruning_trainer import VAEPruningTrainer
from diffusers import AutoencoderTiny

class AutoencoderTinyWrapper(AutoencoderTiny):
    """Wrapper for AutoencoderTiny to make it compatible with our interface"""
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x).clamp(0, 1)
        
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

model = AutoencoderTinyWrapper.from_pretrained("madebyollin/taesd")

trainer = VAEPruningTrainer()

