import torch
import torch.nn as nn
import torch.optim as optim
from torch import inf
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from torch.profiler import profile, record_function, ProfilerActivity

class Bottleneck(nn.Module):
    def __init__(self, in_channels, width, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * 4, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(width * 4)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out

class Resnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(16, 16, 6, 1)
        self.layer2 = self._make_layer(64, 32, 6, 2)
        self.layer3 = self._make_layer(128, 64, 6, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, width, depth, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, width * 4, 2, 2, 0, bias=False),
                nn.BatchNorm2d(width * 4)
            )
        
        elif in_channels != width * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, width * 4, 1, stride, 0, bias=False),
                nn.BatchNorm2d(width * 4)
            )
        
        layers = []
        layers.append(
            Bottleneck(in_channels, width, stride, downsample)
        )
        for _ in range(1, depth):
            layers.append(
                Bottleneck(width * 4, width, 1)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class Dataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class EarlyStopping:
    def __init__(self, patience=10, threshold=1e-4):
        self.patience = patience
        self.threshold = threshold

        self.mode_worse = inf
        self.last_epoch = 0
        self.best = self.mode_worse
        self.num_bad_epochs = 0
        self.early_stop = False

    def __call__(self, metrics):
        current = float(metrics)

        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.early_stop = True


    def is_better(self, a, best):
        rel_epsilon = 1. - self.threshold
        return a < best * rel_epsilon

def calc_error(model, loader, device):
    wrong_pred = 0
    total_pred = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            wrong_pred += (predictions != labels).sum()
            total_pred += predictions.size(0)

        error = 100 * wrong_pred / total_pred
    
    return error

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/cifar10_tc_experiment_b256')

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )

    batch_size = 256

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download = True, transform=None)
    train, val = torch.utils.data.random_split(dataset, [45000, 5000])

    trainset = Dataset(train, transform=train_transform)
    valset = Dataset(val, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    trainloader_test = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)                           

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)

    model = Resnet(3, 100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, threshold=0.025, min_lr=1e-4)
    early_stopping = EarlyStopping(patience=15, threshold=0.025)

    running_loss = 0.0
    iters = 0
    for epoch in range(30):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            iters += 1
            if (iters + 1) % 100 == 0:
                writer.add_scalar('training loss',
                                running_loss / 100,
                                iters + 1)
                print(f'[{epoch + 1}, {iters + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0

                '''
                error = calc_error(model, trainloader_test, device)
                writer.add_scalar('train error',
                            error,
                            iters + 1)
                print(f'Train Error: {error:.3f} %')
                '''

                error = calc_error(model, valloader, device)

                writer.add_scalar('valiation error',
                            error,
                            iters + 1)
                print(f'Validation Error: {error:.3f} %')

                scheduler.step(error)
                early_stopping(error)
                if early_stopping.early_stop:
                    print("stopping:", iters + 1)
                    writer.add_text('early stop', str(iters + 1))
                    break

            if early_stopping.early_stop:
                break

    print('Finished Training')

    error = calc_error(model, testloader, device)
    writer.add_text('test error', str(error))
    print(f'Test Error: {error:.3f} %')

if __name__ == "__main__":
    main()