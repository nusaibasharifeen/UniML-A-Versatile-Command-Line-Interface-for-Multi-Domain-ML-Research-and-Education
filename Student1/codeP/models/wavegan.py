import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
from pathlib import Path
import random
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# class AudioDataset(Dataset):
#     """Custom dataset for loading audio files"""
    
#     def __init__(self, audio_dir, sample_rate=16000, audio_length=16384):
#         self.audio_dir = Path(audio_dir)
#         self.sample_rate = sample_rate
#         self.audio_length = audio_length
        
#         # Get all audio files
#         self.audio_files = []
#         for ext in ['*.wav', '*.mp3', '*.flac']:
#             self.audio_files.extend(self.audio_dir.glob(f"**/{ext}"))
        
#         print(f"Found {len(self.audio_files)} audio files")
        
#     def __len__(self):
#         return len(self.audio_files)
    
#     def __getitem__(self, idx):
#         audio_path = self.audio_files[idx]
        
#         try:
#             # Load audio using torchaudio
#             waveform, sr = torchaudio.load(audio_path)
            
#             # Convert to mono if stereo
#             if waveform.shape[0] > 1:
#                 waveform = torch.mean(waveform, dim=0, keepdim=True)
            
#             # Resample if necessary
#             if sr != self.sample_rate:
#                 resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
#                 waveform = resampler(waveform)
            
#             # Ensure we have the right length
#             if waveform.shape[1] > self.audio_length:
#                 # Random crop
#                 start = random.randint(0, waveform.shape[1] - self.audio_length)
#                 waveform = waveform[:, start:start + self.audio_length]
#             elif waveform.shape[1] < self.audio_length:
#                 # Pad with zeros
#                 pad_length = self.audio_length - waveform.shape[1]
#                 waveform = F.pad(waveform, (0, pad_length))
            
#             # Normalize to [-1, 1]
#             waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
#             return waveform.squeeze(0)  # Remove channel dimension
            
#         except Exception as e:
#             print(f"Error loading {audio_path}: {e}")
#             # Return silence if file can't be loaded
#             return torch.zeros(self.audio_length)

class WaveGANGenerator(nn.Module):
    """WaveGAN Generator Network"""
    
    def __init__(self, latent_dim=100, ngf=64, audio_length=16384):
        super(WaveGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.audio_length = audio_length
        
        # Start with a small feature map and upsample
        self.initial_size = 16
        
        # Initial dense layer
        self.dense = nn.Linear(latent_dim, ngf * 16 * self.initial_size)
        
        # Upsampling layers
        self.conv1 = nn.ConvTranspose1d(ngf * 16, ngf * 8, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(ngf * 8, ngf * 4, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv4 = nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv5 = nn.ConvTranspose1d(ngf, ngf // 2, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv6 = nn.ConvTranspose1d(ngf // 2, ngf // 4, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv7 = nn.ConvTranspose1d(ngf // 4, ngf // 8, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv8 = nn.ConvTranspose1d(ngf // 8, ngf // 16, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv9 = nn.ConvTranspose1d(ngf // 16, ngf // 32, kernel_size=25, stride=2, padding=12, output_padding=1)
        self.conv10 = nn.ConvTranspose1d(ngf // 32, 1, kernel_size=25, stride=2, padding=12, output_padding=1)
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
        # Weight initialization
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        # Dense layer
        x = self.dense(z)
        x = x.view(x.size(0), self.ngf * 16, self.initial_size)
        
        # Progressive upsampling
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.tanh(self.conv10(x))
        
        return x.squeeze(1)

class WaveGANDiscriminator(nn.Module):
    """WaveGAN Discriminator Network"""
    
    def __init__(self, ndf=64, audio_length=16384):
        super(WaveGANDiscriminator, self).__init__()
        self.ndf = ndf
        self.audio_length = audio_length
        
        self.conv1 = nn.Conv1d(1, ndf // 32, kernel_size=25, stride=2, padding=12)
        self.conv2 = nn.Conv1d(ndf // 32, ndf // 16, kernel_size=25, stride=2, padding=12)
        self.conv3 = nn.Conv1d(ndf // 16, ndf // 8, kernel_size=25, stride=2, padding=12)
        self.conv4 = nn.Conv1d(ndf // 8, ndf // 4, kernel_size=25, stride=2, padding=12)
        self.conv5 = nn.Conv1d(ndf // 4, ndf // 2, kernel_size=25, stride=2, padding=12)
        self.conv6 = nn.Conv1d(ndf // 2, ndf, kernel_size=25, stride=2, padding=12)
        self.conv7 = nn.Conv1d(ndf, ndf * 2, kernel_size=25, stride=2, padding=12)
        self.conv8 = nn.Conv1d(ndf * 2, ndf * 4, kernel_size=25, stride=2, padding=12)
        self.conv9 = nn.Conv1d(ndf * 4, ndf * 8, kernel_size=25, stride=2, padding=12)
        self.conv10 = nn.Conv1d(ndf * 8, ndf * 16, kernel_size=25, stride=2, padding=12)
        
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        
        self.final = nn.Linear(ndf * 16 * 16, 1)
        
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.leaky_relu(self.conv6(x))
        x = self.leaky_relu(self.conv7(x))
        x = self.leaky_relu(self.conv8(x))
        x = self.leaky_relu(self.conv9(x))
        x = self.leaky_relu(self.conv10(x))
        
        x = x.view(x.size(0), -1)
        x = self.final(x)
        
        return x

def download_dataset():
    """Download and extract the ESC-50 dataset"""
    dataset_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    dataset_dir = "ESC-50-master"
    
    if not os.path.exists(dataset_dir):
        print("Downloading ESC-50 dataset...")
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            # Download the dataset
            urllib.request.urlretrieve(dataset_url, "esc50.zip")
            
            # Extract the dataset
            with zipfile.ZipFile("esc50.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Remove the zip file
            os.remove("esc50.zip")
            print("ESC-50 dataset downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"Error downloading or extracting ESC-50 dataset: {e}")
            raise Exception("Failed to download or extract ESC-50 dataset")
    
    # Search for the audio directory
    audio_dir = None
    for root, dirs, _ in os.walk(dataset_dir):
        if "audio" in dirs:
            audio_dir = os.path.join(root, "audio")
            break
    
    if audio_dir is None or not os.path.exists(audio_dir):
        print(f"Directory structure in {dataset_dir}:")
        for root, dirs, files in os.walk(dataset_dir):
            print(f"Root: {root}, Dirs: {dirs}, Files: {len(files)}")
        raise Exception("ESC-50 audio directory not found")
    
    # Verify that audio files exist
    audio_files = list(Path(audio_dir).glob("**/*.wav"))
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        raise Exception("No audio files found in ESC-50 dataset")
    
    print(f"Found audio directory: {audio_dir}")
    print(f"Number of audio files: {len(audio_files)}")
    
    return audio_dir

# def train_wavegan(generator, discriminator, dataloader, num_epochs=100, lr=0.0001):
#     """Train WaveGAN"""
    
#     criterion = nn.BCEWithLogitsLoss()
    
#     optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
#     generator.train()
#     discriminator.train()
    
#     fixed_noise = torch.randn(16, generator.latent_dim).to(device)
    
#     G_losses = []
#     D_losses = []
    
#     for epoch in range(num_epochs):
#         epoch_G_loss = 0
#         epoch_D_loss = 0
        
#         progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
#         for i, real_audio in enumerate(progress_bar):
#             batch_size = real_audio.size(0)
#             real_audio = real_audio.to(device)
            
#             real_labels = torch.ones(batch_size, 1).to(device)
#             fake_labels = torch.zeros(batch_size, 1).to(device)
            
#             optimizer_D.zero_grad()
            
#             output_real = discriminator(real_audio)
#             d_loss_real = criterion(output_real, real_labels)
            
#             noise = torch.randn(batch_size, generator.latent_dim).to(device)
#             fake_audio = generator(noise)
#             output_fake = discriminator(fake_audio.detach())
#             d_loss_fake = criterion(output_fake, fake_labels)
            
#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_D.step()
            
#             optimizer_G.zero_grad()
            
#             output_fake = discriminator(fake_audio)
#             g_loss = criterion(output_fake, real_labels)
            
#             g_loss.backward()
#             optimizer_G.step()
            
#             epoch_G_loss += g_loss.item()
#             epoch_D_loss += d_loss.item()
            
#             progress_bar.set_postfix({
#                 'G_loss': f'{g_loss.item():.4f}',
#                 'D_loss': f'{d_loss.item():.4f}'
#             })
        
#         avg_G_loss = epoch_G_loss / len(dataloader)
#         avg_D_loss = epoch_D_loss / len(dataloader)
        
#         G_losses.append(avg_G_loss)
#         D_losses.append(avg_D_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}] - G_loss: {avg_G_loss:.4f}, D_loss: {avg_D_loss:.4f}')
        
#         if (epoch + 1) % 10 == 0:
#             generator.eval()
#             with torch.no_grad():
#                 fake_samples = generator(fixed_noise)
#                 sample_audio = fake_samples[0].cpu().numpy()
                
#                 os.makedirs('generated_samples', exist_ok=True)
                
#                 torchaudio.save(
#                     f'generated_samples/sample_epoch_{epoch+1}.wav',
#                     torch.tensor(sample_audio).unsqueeze(0),
#                     16000
#                 )
#             generator.train()
    
#     return G_losses, D_losses

# def generate_samples(generator, num_samples=5):
#     """Generate audio samples"""
#     generator.eval()
    
#     os.makedirs('final_samples', exist_ok=True)
    
#     with torch.no_grad():
#         for i in range(num_samples):
#             noise = torch.randn(5, generator.latent_dim).to(device)
#             fake_audio = generator(noise)
            
#             audio_np = fake_audio[0].cpu().numpy()
            
#             torchaudio.save(
#                 f'final_samples/generated_sample_{i+1}.wav',
#                 torch.tensor(audio_np).unsqueeze(0),
#                 16000
#             )
            
#             print(f"Generated sample {i+1} saved")

# def main():
#     """Main training function"""
    
#     BATCH_SIZE = 16
#     LATENT_DIM = 100
#     AUDIO_LENGTH = 16384
#     NUM_EPOCHS = 50
#     LEARNING_RATE = 0.0001
    
#     # Download dataset
#     dataset_dir = download_dataset()
    
#     # Create dataset and dataloader
#     dataset = AudioDataset(dataset_dir, audio_length=AUDIO_LENGTH)
    
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
#     # Initialize models
#     generator = WaveGANGenerator(
#         latent_dim=LATENT_DIM,
#         audio_length=AUDIO_LENGTH
#     ).to(device)
    
#     discriminator = WaveGANDiscriminator(
#         audio_length=AUDIO_LENGTH
#     ).to(device)
    
#     # Test model architectures
#     print("\nTesting model architectures...")
#     test_batch_size = 2
#     test_noise = torch.randn(test_batch_size, LATENT_DIM).to(device)
#     test_audio = torch.randn(test_batch_size, AUDIO_LENGTH).to(device)
    
#     try:
#         gen_output = generator(test_noise)
#         print(f"Generator output shape: {gen_output.shape}")
        
#         disc_output = discriminator(test_audio)
#         print(f"Discriminator output shape: {disc_output.shape}")
        
#         disc_gen_output = discriminator(gen_output)
#         print(f"Discriminator on generated audio shape: {disc_gen_output.shape}")
        
#         print("✓ Model architectures are working correctly!")
        
#     except Exception as e:
#         print(f"✗ Error in model architecture: {e}")
#         return
    
#     # Train the model
#     print(f"\nStarting training for {NUM_EPOCHS} epochs...")
#     G_losses, D_losses = train_wavegan(
#         generator, discriminator, dataloader,
#         num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
#     )

    
#     # Generate final samples
#     print("\nGenerating final samples...")
#     generate_samples(generator, num_samples=5)
    
#     # Save models
#     torch.save(generator.state_dict(), 'wavegan_generator.pth')
#     torch.save(discriminator.state_dict(), 'wavegan_discriminator.pth')
    
#     print("Training completed! Models saved.")
#     print("Generated samples are available in the 'final_samples' directory.")

# if __name__ == "__main__":
#     main()