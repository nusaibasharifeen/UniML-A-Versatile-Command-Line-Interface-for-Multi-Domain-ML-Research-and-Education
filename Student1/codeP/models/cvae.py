import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Encoder(nn.Module):
    """
    Conditional Encoder Network
    Takes image and condition label as input
    Outputs mean and log variance of latent distribution
    """
    def __init__(self, num_classes=10, img_channels=1, img_size=28, latent_dim=20, hidden_dim=256):
        super(Encoder, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Input dimension: flattened image + embedded label
        input_dim = img_channels * img_size * img_size + num_classes

        # Encoder layers
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(0.2),

            # Second layer
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),

            # Third layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)

        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate image and embedded labels
        enc_input = torch.cat([img_flat, label_embed], dim=1)

        # Encode
        h = self.model(enc_input)

        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

class Decoder(nn.Module):
    """
    Conditional Decoder Network
    Takes latent vector and condition label as input
    Outputs reconstructed image conditioned on the label
    """
    def __init__(self, num_classes=10, img_channels=1, img_size=28, latent_dim=20, hidden_dim=256):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Input dimension: latent vector + embedded label
        input_dim = latent_dim + num_classes

        # Decoder layers
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),

            # Second layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),

            # Third layer
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 4),

            # Output layer
            nn.Linear(hidden_dim * 4, img_channels * img_size * img_size),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, z, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate latent vector and embedded labels
        dec_input = torch.cat([z, label_embed], dim=1)

        # Decode
        img = self.model(dec_input)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)

        return img

class ConditionalVAE:
    """
    Complete Conditional VAE implementation with training utilities
    """
    def __init__(self, num_classes=10, img_channels=1, img_size=28, latent_dim=20,
                 lr=0.001, beta=1.0):
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.beta = beta  # Beta for KL divergence weighting

        # Initialize networks
        self.encoder = Encoder(num_classes, img_channels, img_size, latent_dim).to(self.device)
        self.decoder = Decoder(num_classes, img_channels, img_size, latent_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )

        # Training history
        self.total_losses = []
        self.recon_losses = []
        self.kl_losses = []

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function: reconstruction loss + KL divergence"""
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss, kl_loss

    def train_step(self, real_imgs, labels):
        """Single training step"""
        self.optimizer.zero_grad()

        # Encode
        mu, logvar = self.encoder(real_imgs, labels)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_imgs = self.decoder(z, labels)

        # Calculate losses
        recon_loss, kl_loss = self.loss_function(recon_imgs, real_imgs, mu, logvar)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), recon_loss.item(), kl_loss.item()

    def train(self, dataloader, epochs=100, save_interval=10):
        """Complete training procedure"""
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0

            # Training loop
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for i, (imgs, labels) in enumerate(pbar):
                # Move to device
                real_imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Train step
                total_loss, recon_loss, kl_loss = self.train_step(real_imgs, labels)

                epoch_total_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'Total_loss': f'{total_loss:.4f}',
                    'Recon_loss': f'{recon_loss:.4f}',
                    'KL_loss': f'{kl_loss:.4f}'
                })

            # Record average losses
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches

            self.total_losses.append(avg_total_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)

            print(f"Epoch {epoch+1}/{epochs} - Total: {avg_total_loss:.4f}, "
                  f"Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")

            # Save sample images
            if (epoch + 1) % save_interval == 0:
                self.save_sample_images(epoch + 1)
                self.save_reconstructions(dataloader, epoch + 1)

    def generate_samples(self, num_samples=10, specific_class=None):
        """Generate sample images from random latent vectors"""
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim, device=self.device)

            # Generate labels
            if specific_class is not None:
                labels = torch.full((num_samples,), specific_class, device=self.device, dtype=torch.long)
            else:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)

            # Generate images
            gen_imgs = self.decoder(z, labels)

        self.encoder.train()
        self.decoder.train()
        return gen_imgs, labels

    def reconstruct_images(self, imgs, labels):
        """Reconstruct images through encoder-decoder"""
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(imgs, labels)

            # Use mean for reconstruction (no sampling)
            z = mu

            # Decode
            recon_imgs = self.decoder(z, labels)

        self.encoder.train()
        self.decoder.train()
        return recon_imgs

    def save_sample_images(self, epoch, num_samples=25):
        """Save a grid of sample images"""
        gen_imgs, labels = self.generate_samples(num_samples)

        # Create directory if it doesn't exist
        os.makedirs('cvae_samples', exist_ok=True)

        # Save image grid
        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, normalize=True)
        results_dir = os.path.join("RESULTS", "cvae")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'cvae_samples/generated_epoch_{epoch}.png')
        torchvision.utils.save_image(grid, save_path)

    def save_reconstructions(self, dataloader, epoch, num_samples=10):
        """Save comparison of original and reconstructed images"""
        # Get a batch of real images
        real_imgs, labels = next(iter(dataloader))
        real_imgs = real_imgs[:num_samples].to(self.device)
        labels = labels[:num_samples].to(self.device)

        # Reconstruct images
        recon_imgs = self.reconstruct_images(real_imgs, labels)

        # Create comparison grid
        comparison = torch.cat([real_imgs, recon_imgs], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True)

        # Save comparison
        results_dir = os.path.join("RESULTS", "cvae")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'cvae_samples/reconstruction_epoch_{epoch}.png')
        torchvision.utils.save_image(grid, save_path)

    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.total_losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.recon_losses, label='Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.kl_losses, label='KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL Divergence Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def generate_class_samples(self, save_path):
        """Generate samples for each class"""
        samples_per_class = 5
        all_imgs = []

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for class_idx in range(self.num_classes):
                z = torch.randn(samples_per_class, self.latent_dim, device=self.device)
                labels = torch.full((samples_per_class,), class_idx, device=self.device, dtype=torch.long)
                gen_imgs = self.decoder(z, labels)
                all_imgs.append(gen_imgs)

        # Concatenate all images
        all_imgs = torch.cat(all_imgs, dim=0)

        # Create grid
        grid = torchvision.utils.make_grid(all_imgs, nrow=samples_per_class, normalize=True)
        torchvision.utils.save_image(grid, save_path)

        self.encoder.train()
        self.decoder.train()
        print(f"Class samples saved to {save_path}")

    def interpolate_in_latent_space(self, class1, class2, num_steps=10, save_path='cvae_interpolation.png'):
        """Interpolate between two classes in latent space"""
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # Generate latent vectors for both classes
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)

            # Interpolate between latent vectors
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            interpolated_imgs = []

            for alpha in alphas:
                # Interpolate latent vector
                z_interp = (1 - alpha) * z1 + alpha * z2

                # Interpolate class (choose based on alpha)
                if alpha < 0.5:
                    label = torch.tensor([class1], device=self.device, dtype=torch.long)
                else:
                    label = torch.tensor([class2], device=self.device, dtype=torch.long)

                # Generate image
                img = self.decoder(z_interp, label)
                interpolated_imgs.append(img)

            # Concatenate all images
            all_imgs = torch.cat(interpolated_imgs, dim=0)

            # Create grid
            grid = torchvision.utils.make_grid(all_imgs, nrow=num_steps, normalize=True)
            torchvision.utils.save_image(grid, save_path)

        self.encoder.train()
        self.decoder.train()
        print(f"Interpolation saved to {save_path}")

