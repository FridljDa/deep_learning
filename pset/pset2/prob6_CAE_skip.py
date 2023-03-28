
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
#import vutils


# DESCRIBing THE CNN ARCHITECTURE 
class CAE_skip(nn.Module):
    def __init__(self):
        super(CAE_skip, self).__init__()
        # Building an linear encoder with Linear
        #encoder
        self.conv1 = nn.Conv2d(1, 4, kernel_size = 3, stride=1, padding = 1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size = 3, stride=1, padding = 1)
        self.fc1 = nn.Linear(8*7*7, 20)
        self.fc2 = nn.Linear(20, 2)
        #decoder
        self.fc3 = nn.Linear(2, 20)
        self.fc4 = nn.Linear(20, 8*7*7)
        self.conv3 = nn.Conv2d(8, 4, kernel_size = 3, stride=1, padding = 1)
        self.conv4 = nn.Conv2d(4, 1, kernel_size = 3, stride=1, padding = 1)
       
    
    def forward(self, x0):
        x10, x4, x1 = self.encoder(x0)
        x = self.decoder(x10, x4, x1)

        return x
    
    def encoder(self, x0):
        x1 = self.conv1(x0)#
        x2 = F.relu(x1)
        x3 = F.max_pool2d(x2, 2, 2)
        x4 = self.conv2(x3)#
        x5 = F.relu(x4)
        x6 = F.max_pool2d(x5, 2, 2)
        x7 = x6.view(-1,8*7*7)
        x8 = self.fc1(x7)
        x9 = F.relu(x8)
        x10 = self.fc2(x9)
        return x10, x4, x1
    
    def decoder(self, x10, x4, x1):
        x11 = self.fc3(x10)
        x12 = F.relu(x11)
        x13 = self.fc4(x12)
        x14 = F.relu(x13)
        x15 = x14.view(-1,8,7,7)
        x16 = nn.Upsample(scale_factor=2, mode='nearest')(x15) + x4
        x17 = F.relu(self.conv3(x16))
        x18 = nn.Upsample(scale_factor=2, mode='nearest')(x17)+ x1
        x19 = self.conv4(x18)
        return x19
    
def train_log(model, device, train_loader, optimizer, epoch, log_interval = 10):

    # ANNOTATION 1: Enter train mode
    model.train()
    train_loss = 0
    loss_func = torch.nn.MSELoss()

    # ANNOTATION 2: Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.reshape(-1, 28*28)
        #data = torch.flatten(data, start_dim=1)

        # ANNOTATION 3: Reset gradients
        optimizer.zero_grad()
        output = model(data)

        #output = torch.flatten(output, start_dim=1)

        # ANNOTATION 4: Calculate the loss
        loss = loss_func(data, output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')

    train_loss /= len(train_loader)
    print(f"\rTrain Epoch: {epoch: 3} \t| Train set: Average loss: {train_loss:.4f}{' '*50}")

def train(args, model, device, train_loader, optimizer, epoch):

    # ANNOTATION 1: Enter train mode
    model.train()
    train_loss = 0
    loss_func = torch.nn.MSELoss()

    # ANNOTATION 2: Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.reshape(-1, 28*28)
        #data = torch.flatten(data, start_dim=1)

        # ANNOTATION 3: Reset gradients
        optimizer.zero_grad()
        output = model(data)

        #output = torch.flatten(output, start_dim=1)

        # ANNOTATION 4: Calculate the loss
        loss = loss_func(data, output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')

    train_loss /= len(train_loader)
    print(f"\rTrain Epoch: {epoch: 3} \t| Train set: Average loss: {train_loss:.4f}{' '*50}")

def test(model, device, test_loader, epoch):

    model.eval()
    test_loss = 0
    #correct = 0
    loss_func = torch.nn.MSELoss(reduction='sum')

    # stop tracking gradients
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            #data = data.reshape(-1, 28*28)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #compare data to output
            test_loss += loss_func(data, output).item()
            #print(test_loss)

    
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}\n'.format(
        test_loss))

def main_wrapper(train_loader, test_loader, model, file_name):
    # Training settings
    args =  dict(batch_size=64, 
        test_batch_size=1000, 
        epochs=10, 
        momentum=0.5,
        no_cuda=False,
        seed=1,
        lr = 0.001,
        log_interval=10,
        save_model=True)
    
   
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()

    torch.manual_seed(args["seed"])

    device = torch.device("cuda" if use_cuda else "cpu")
   
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=args["lr"],
                             weight_decay = 1e-8)
    
    for epoch in range(1, args["epochs"] + 1):
        train_log(model, device, train_loader, optimizer, epoch, log_interval = 10)
        test(model, device, test_loader, epoch)

    if (args["save_model"]):
        torch.save(model.state_dict(), file_name)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default= 0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Download the MNIST dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # training data
    ntrain = 60000
    train_data = (mnist_trainset.data.to(dtype=torch.float32)[:ntrain]/255).view(-1, 1, 28, 28)
    train_labels = mnist_trainset.targets.to(dtype=torch.long)[:ntrain]

    # testing data
    ntest = 2000
    test_data = (mnist_testset.data.to(dtype=torch.float32)[:ntest]/255).view(-1, 1, 28, 28)
    test_labels = mnist_testset.targets.to(dtype=torch.long)[:ntest]

    # Load into torch datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, drop_last=True, shuffle=True, **kwargs
    )

    model = CAE_skip().to(device)

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay = 1e-8)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cae_skip.pt")


if __name__ == '__main__':
    main()
