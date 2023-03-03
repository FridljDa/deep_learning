
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



# from einops import rearrange, reduce
import matplotlib.pyplot as plt

# DESCRIBing THE CNN ARCHITECTURE 
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Building an linear encoder with Linear
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 4, kernel_size = 3, stride=1, padding = 1),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Conv2d(4, 8, kernel_size = 3, stride=1, padding = 1),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Flatten(),
            torch.nn.Linear(8*7*7, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 8*7*7),
            torch.nn.ReLU(),
            nn.Unflatten(1, (8,7,7)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 4, kernel_size = 3, stride=1, padding = 1),
            torch.nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(4, 1, kernel_size = 3, stride=1, padding = 1),
            torch.nn.Sigmoid()
        )
       
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    #def encode(self, x):
    #    x = F.relu(self.conv1(x)) #7
    #    x = F.max_pool2d(x, 2, 2) #6
    #    x = F.relu(self.conv2(x)) #5
    #    x = F.max_pool2d(x, 2, 2) #4
    #    x = x.view(-1,4*4*80) #3
    #    x = F.relu(self.fc1(x)) #2 
    #    x = self.fc2(x) #1

    #    return x
    
    #def decode(self, x):
     #   x = F.relu(self.fc3(x)) #1
     #   x = F.relu(self.fc4(x)) #2
     #   x = x.view(-1,80,4,4) #3
    #    x = F.max_unpool2d(x, 2, 2) #4

    #    x = F.relu(self.conv3(x)) #5
    #    x = F.max_unpool2d(x, 2, 2) #6
        
    #    x = self.conv4(x) #7

    #    return x
    
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

def test(args, model, device, test_loader, epoch):

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

    #TODO: fix this
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}\n'.format(
        test_loss))
    
    #in the last epoch, plot original and reconstructed images
    if epoch == args.epochs:
        #plot original and reconstructed images

        #take first image from batch
        data = data[2]
        output = output[2]
        #reshape to 28x28
        data = data.reshape(28,28)
        output = output.reshape(28,28)
        #plot data and output next to each other

        #plot data and output next to each other
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(data, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(output, cmap='gray')
        #plt.show()
        #save plot
        plt.savefig('reconstructed_images.png')


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

    model = CAE().to(device)

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay = 1e-8)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cae.pt")


if __name__ == '__main__':
    main()
