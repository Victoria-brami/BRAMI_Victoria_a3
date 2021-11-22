import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from models.modeling import VisionTransformer, CONFIGS
import numpy as np

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


def get_model(model_name, checkpoint=None, trained=True, in_size=224):

    input_size = in_size # For resnet

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=trained)
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
        model = models.resnet34(pretrained=trained)
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
        model = models.resnet50(pretrained=trained)
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
        model = models.resnet101(pretrained=trained)
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=2048, out_features=4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.PReLU(),
            nn.Linear(in_features=4096, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        # model.load_state_dict(torch.load('../resnet152-b121ed2d.pth'))
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.BatchNorm1d(1924),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )

    elif model_name == 'vit-B_16':
        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, input_size, zero_head=True, num_classes=num_classes)
        model.load_from(np.load('../pretrained_models/ViT-B_16.npz'))

    elif model_name == 'vit-B_16-224':
        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, input_size, zero_head=True, num_classes=num_classes)
        model.load_from(np.load('../pretrained_models/ViT-B_16-224.npz'))

    elif model_name == 'vit-L_16':
        config = CONFIGS['ViT-L_16']
        model = VisionTransformer(config, input_size, zero_head=True, num_classes=num_classes)
        model.load_from(np.load('../pretrained_models/ViT-L_16.npz'))
    
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=trained)
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

def create_model_for_evaluation(model_name, checkpoint_filename, use_cuda=False):

    if model_name == 'resnet152':
        model = models.resnet152(pretrained=False)
        # model.load_state_dict(torch.load('../resnet152-b121ed2d.pth'))
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )
        if use_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'
        checkpoint = torch.load(checkpoint_filename, map_location=device)
        for key in checkpoint.keys():
            print(key)
        model.load_state_dict(checkpoint)

        return model

    elif model_name == 'vit-B_16-384':
        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 384, zero_head=True, num_classes=num_classes)
        #model.load_from(np.load(checkpoint_filename))
        model.load_state_dict(torch.load(checkpoint_filename))
        return model

    elif model_name == 'vit-L_16-384':
        config = CONFIGS['ViT-L_16']
        model = VisionTransformer(config, 384, zero_head=True, num_classes=num_classes)
        model.load_state_dict(torch.load(checkpoint_filename))
        return model


if __name__ == '__main__':
    channels = 3
    H = 224
    W = 224
    net = Vi
    summary(net, input_size=(channels, H, W))
