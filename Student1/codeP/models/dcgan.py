import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN:
    def __init__(self, nz=100, lr=0.0002, beta1=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.nz = nz
        self.generator = Generator(nz=nz).to(device)
        self.discriminator = Discriminator().to(device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        self.G_losses = []
        self.D_losses = []
        self.img_list = []

    def train(self, dataloader, num_epochs=5):
        real_label = 1.0
        fake_label = 0.0
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                self.discriminator.zero_grad()
                real_batch = data[0].to(self.device)
                b_size = real_batch.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.discriminator(real_batch).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizer_D.step()

                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizer_G.step()

                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                if i % 100 == 0:
                    print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                          f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                          f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

                if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    def generate_samples(self, num_samples=64):
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            return self.generator(noise).detach().cpu()
