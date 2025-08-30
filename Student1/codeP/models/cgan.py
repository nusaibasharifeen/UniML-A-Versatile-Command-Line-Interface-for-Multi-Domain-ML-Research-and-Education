import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

class Generator(nn.Module):
    """
    Conditional Generator Network
    Takes noise vector z and condition label c as input
    Outputs generated image conditioned on the label
    """
    def __init__(self, noise_dim=100, num_classes=10, img_channels=1, img_size=28, hidden_dim=256):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Input dimension: noise + embedded label
        input_dim = noise_dim + num_classes

        # Generator layers
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),

            # Second layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),

            # Third layer
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 4),

            # Output layer
            nn.Linear(hidden_dim * 4, img_channels * img_size * img_size),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, label_embed], dim=1)

        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)

        return img

class Discriminator(nn.Module):
    """
    Conditional Discriminator Network
    Takes image and condition label as input
    Outputs probability that the image is real and matches the condition
    """
    def __init__(self, num_classes=10, img_channels=1, img_size=28, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Input dimension: flattened image + embedded label
        input_dim = img_channels * img_size * img_size + num_classes

        # Discriminator layers
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Second layer
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Third layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)

        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate image and embedded labels
        disc_input = torch.cat([img_flat, label_embed], dim=1)

        # Get discriminator output
        validity = self.model(disc_input)

        return validity

class ConditionalGAN:
    """
    Complete Conditional GAN implementation with training utilities
    """
    def __init__(self, noise_dim=100, num_classes=10, img_channels=1, img_size=28,
                 lr=0.0002, beta1=0.5, beta2=0.999):
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.generator = Generator(noise_dim, num_classes, img_channels, img_size).to(self.device)
        self.discriminator = Discriminator(num_classes, img_channels, img_size).to(self.device)

        # Loss function
        self.adversarial_loss = nn.BCELoss()

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        # Training history
        self.g_losses = []
        self.d_losses = []

    def train_step(self, real_imgs, labels, batch_size):
        """Single training step"""

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=self.device, requires_grad=False)
        fake = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)

        # ---------------------
        #  Train Generator
        # ---------------------

        self.optimizer_G.zero_grad()

        # Sample noise and labels for generator
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)

        # Generate fake images
        gen_imgs = self.generator(z, gen_labels)

        # Generator loss: fool discriminator
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, gen_labels), valid)

        g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Real images loss
        real_loss = self.adversarial_loss(self.discriminator(real_imgs, labels), valid)

        # Fake images loss
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), gen_labels), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        return g_loss.item(), d_loss.item()

    def train(self, dataloader, epochs=100, save_interval=10):
        """Complete training procedure"""

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0

            # Training loop
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for i, (imgs, labels) in enumerate(pbar):
                batch_size = imgs.shape[0]

                # Move to device
                real_imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Train step
                g_loss, d_loss = self.train_step(real_imgs, labels, batch_size)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}'
                })

            # Record average losses
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            print(f"Epoch {epoch+1}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")

    def generate_samples(self, num_samples=10, specific_class=None):
        """Generate sample images"""
        self.generator.eval()

        with torch.no_grad():
            # Generate noise
            z = torch.randn(num_samples, self.noise_dim, device=self.device)

            # Generate labels
            if specific_class is not None:
                labels = torch.full((num_samples,), specific_class, device=self.device, dtype=torch.long)
            else:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)

            # Generate images
            gen_imgs = self.generator(z, labels)

        self.generator.train()
        return gen_imgs, labels

    def generate_class_samples(self, save_path='class_samples.png'):
        """Generate one sample for each class"""
        samples_per_class = 5
        all_imgs = []

        self.generator.eval()
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                z = torch.randn(samples_per_class, self.noise_dim, device=self.device)
                labels = torch.full((samples_per_class,), class_idx, device=self.device, dtype=torch.long)
                gen_imgs = self.generator(z, labels)
                all_imgs.append(gen_imgs)

        # Concatenate all images
        all_imgs = torch.cat(all_imgs, dim=0)
        all_imgs = (all_imgs + 1) / 2  # Denormalize

        # Create grid
        grid = torchvision.utils.make_grid(all_imgs, nrow=samples_per_class, normalize=True)
        torchvision.utils.save_image(grid, save_path)

        self.generator.train()
        print(f"Class samples saved to {save_path}")