import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_CIFAR10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_data, test_data