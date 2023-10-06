import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_apis

# Define a custom linear function with explicit weights and biases
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t()) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias

# Define a LeNet-300-100-like neural network
class LeNet300100(nn.Module):
    def __init__(self, in_features, hidden1_features, hidden2_features, out_features):
        super(LeNet300100, self).__init__()

        self.weight1 = nn.Parameter(torch.rand(in_features, hidden1_features))
        self.bias1 = nn.Parameter(torch.rand(hidden1_features))
        self.weight2 = nn.Parameter(torch.rand(hidden1_features, hidden2_features))
        self.bias2 = nn.Parameter(torch.rand(hidden2_features))
        self.weight3 = nn.Parameter(torch.rand(hidden2_features, out_features))
        self.bias3 = nn.Parameter(torch.rand(out_features))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.matmul(x, self.weight1) + self.bias1.unsqueeze(0)
        x = self.relu(x)
        x = torch.matmul(x, self.weight2) + self.bias2.unsqueeze(0)
        x = self.relu(x)
        x = torch.matmul(x, self.weight3) + self.bias3.unsqueeze(0)
        return x

# Define a LeNet-300-100-like neural network
class CustomLeNet300100(nn.Module):
    def __init__(self, in_features, hidden1_features, hidden2_features, out_features):
        super(CustomLeNet300100, self).__init__()

        self.weight1 = nn.Parameter(torch.rand(in_features, hidden1_features))
        self.bias1 = nn.Parameter(torch.rand(hidden1_features))
        self.weight2 = nn.Parameter(torch.rand(hidden1_features, hidden2_features))
        self.bias2 = nn.Parameter(torch.rand(hidden2_features))
        self.weight3 = nn.Parameter(torch.rand(hidden2_features, out_features))
        self.bias3 = nn.Parameter(torch.rand(out_features))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = pytorch_apis.customlinear(x, self.weight1, self.bias1, x.device)
        x = self.relu(x)
        x = pytorch_apis.customlinear(x, self.weight2, self.bias2, x.device)
        x = self.relu(x)
        x = pytorch_apis.customlinear(x, self.weight3, self.bias3, x.device)
        return x

def run_guy(model, x_train):
    torch.manual_seed(101)
    # Create a toy dataset
    y_train = x_train.clone()

    # Define the model and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 2000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch == num_epochs // 2):
            optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Test the trained model
    x_test = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    predicted = model(x_test)
    print(f"Predicted: {predicted.detach().numpy()}")

    return model.weight1.detach().numpy(),model.weight2.detach().numpy(),model.weight3.detach().numpy()

torch.manual_seed(101)
model = LeNet300100(in_features=3, hidden1_features=6, hidden2_features=4, out_features=3)
x_train = ((torch.rand(1000, 3)) * 10.0) + 1.0

w1, w2, w3 = run_guy(model, x_train)
print(w1)