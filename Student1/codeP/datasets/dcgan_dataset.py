import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./DATA/dcgan_dataset/training/data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    return dataloader

