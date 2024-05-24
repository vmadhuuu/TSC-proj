import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, n_in, n_classes):
        super(ConvNet, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(1, 128, 7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, self.n_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, self.n_in)  # Reshape input to (batch_size, 1, n_in)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.avg_pool1d(x, 2)  # Use 1D pooling

        x = torch.mean(x, dim=2)  # Average over the length dimension

        x = x.view(-1, 128)  # Flatten the tensor for the fully connected layer

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
