import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    data_train = datasets.MNIST(root='./data', transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root='./data', transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=32, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=32, shuffle=False)

    for data in data_train:
        X_train, y_train = data
        print(X_train)                                                                                                                                                                                                                                                                                                import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    data_train = datasets.MNIST(root='./data', transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root='./data', transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=32, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=32, shuffle=False)

    for data in data_train:
        X_train, y_train = data
        print(X_train)