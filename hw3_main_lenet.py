import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_apis
import torchvision

# Base non-custom lenet for comparison
class LeNet300100(nn.Module):
    def __init__(self, in_dim, hidden1_features, hidden2_features, out_features):
        super(LeNet300100, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        final_dim = ((((in_dim - 2) // 2) - 2) // 2)
        self.fc_in_size = 64 * final_dim * final_dim
        self.weight1 = nn.Parameter(torch.rand(self.fc_in_size, hidden1_features))
        self.bias1 = nn.Parameter(torch.rand(1, hidden1_features))
        self.weight2 = nn.Parameter(torch.rand(hidden1_features, hidden2_features))
        self.bias2 = nn.Parameter(torch.rand(1, hidden2_features))
        self.weight3 = nn.Parameter(torch.rand(hidden2_features, out_features))
        self.bias3 = nn.Parameter(torch.rand(1, out_features))
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.weight1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight2, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight3, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias2, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias3, mode='fan_in', nonlinearity='relu')

        self.fc1 = nn.Linear(64 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_in_size)

        x = torch.matmul(x, self.weight1) + self.bias1
        x = self.relu(x)
        x = torch.matmul(x, self.weight2) + self.bias2
        x = self.relu(x)
        x = torch.matmul(x, self.weight3) + self.bias3
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        '''
        return x

# Lenet-300-100 implemented with custom cuda linear layers
class CustomLeNet300100(nn.Module):
    def __init__(self, in_dim, hidden1_features, hidden2_features, out_features):
        super(CustomLeNet300100, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        final_dim = ((((in_dim - 2) // 2) - 2) // 2)
        self.fc_in_size = 64 * final_dim * final_dim
        self.weight1 = nn.Parameter(torch.rand(self.fc_in_size, hidden1_features))
        self.bias1 = nn.Parameter(torch.rand(1, hidden1_features))
        self.weight2 = nn.Parameter(torch.rand(hidden1_features, hidden2_features))
        self.bias2 = nn.Parameter(torch.rand(1, hidden2_features))
        self.weight3 = nn.Parameter(torch.rand(hidden2_features, out_features))
        self.bias3 = nn.Parameter(torch.rand(1, out_features))
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.weight1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight2, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight3, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias2, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bias3, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_in_size)

        x = pytorch_apis.customlinear(x, self.weight1, self.bias1.squeeze(), x.device)
        x = self.relu(x)
        x = pytorch_apis.customlinear(x, self.weight2, self.bias2.squeeze(), x.device)
        x = self.relu(x)
        x = pytorch_apis.customlinear(x, self.weight3, self.bias3.squeeze(), x.device)
        return x

def load_mnist(batch_size):
    train_trans = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.RandomRotation(degrees=(-25, 25)),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,)) # taken from online recomendation
                            ])

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./DATA/', train=True, download=True,
                                transform=train_trans),
    batch_size=batch_size, shuffle=True,  num_workers=2, persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./DATA/', train=False, download=True,
                                transform=train_trans),
    batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    return train_loader, test_loader


def train_model(model, train_data, test_data):
    torch.manual_seed(101)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 10

    model.cuda()
    
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss}')

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test: {100 * correct / total}%')


torch.manual_seed(101)
model = CustomLeNet300100(in_dim=28, hidden1_features=300, hidden2_features=100, out_features=10)
train_data, test_data = load_mnist(256)
train_model(model, train_data, test_data)
