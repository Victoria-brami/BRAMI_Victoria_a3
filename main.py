import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

torch.cuda.empty_cache()

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='../bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='adam', metavar='O',
                    help='Chosen optimizer')
parser.add_argument('--scheduler', type=str, default='exponential')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--model_name', type=str, default='resnet18', metavar='R',
                    help='Name of the model used.')
parser.add_argument('--in_size', type=int)
parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='W')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint filename.')
parser.add_argument('--aug_type', nargs='+', default=['flip', 'colors', 'rotate', 'erasing'], help='Checkpoint filename.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
val_use_cuda = True
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import *

### ADDED

### END ADDED

data_transforms = build_augmentation(args.aug_type, input_size=args.in_size)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=default_data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import *
model, _ = get_model(args.model_name, checkpoint=args.checkpoint, trained=True, in_size=args.in_size)
if use_cuda:
    print('Using GPU')
    model.cuda()
if val_use_cuda:
    print('Using GPU for validation')
    model.cuda()
else:
    print('Using CPU')

if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.scheduler == 'cosine':    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
elif args.scheduler == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
else:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

filename = '{}_yolo_b{}_lr_{}_sched_{}_scores.txt'.format(args.model_name, args.batch_size, args.lr, args.scheduler)

def train(epoch, txt_file=filename):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if 'vit' in args.model_name:
            output = output[0]
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item())
                )
            filetxt = open(txt_file, 'a+')
            filetxt.write('\n  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100. * batch_idx /len(train_loader), loss.data.item()))
            filetxt.close()
    scheduler.step()       


def validation(txt_file=filename):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if val_use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        if 'vit' in args.model_name:
            output = output[0]
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    ### ADDED
    filetxt = open(txt_file, 'a+')
    filetxt.write('\n   Epoch {}:   Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    filetxt.close()

    acc = correct / len(val_loader.dataset)
    return acc
    ### End ADDED

if __name__ == '__main__':
    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    filetxt = open(filename, 'a+')
    json.dump(vars(args), filetxt, indent=4)
    filetxt.close()
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val_acc = validation()
        model_file = args.experiment + '/{}_sgd_{}_model_'.format(args.model_name, args.in_size) + str(epoch) + '.pth'
        if best_acc <= val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
