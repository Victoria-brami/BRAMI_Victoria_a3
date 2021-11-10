import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

num_classes = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(model_name, checkpoint=None, pretrained=True):

    input_size = 224 # For resnet

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )
    else:
        model = Net()
    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    return model, input_size

if __name__ == '__main__':
    channels = 3
    H = 224
    W = 224
    net = get_model('resnet18', None, False)
    summary(net, input_size=(channels, H, W))