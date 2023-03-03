
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

# from einops import rearrange, reduce
import matplotlib.pyplot as plt

def hello_world():
    print("Hello World")

# DESCRIBing THE CNN ARCHITECTURE 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 80, 5, 1)
        self.fc1 = nn.Linear(4*4*80, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,4*4*80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):

    # ANNOTATION 1: Enter train mode
    model.train()
    train_loss = 0

    # ANNOTATION 2: Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # ANNOTATION 3: Reset gradients
        optimizer.zero_grad()
        output = model(data)

        # ANNOTATION 4: Calculate the loss
        loss = F.nll_loss(output, target)
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
    correct = 0

    # stop tracking gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #############################################
    # TODO: on final epoch, extract filters from model.conv1 and save them 
    # as an image. 
    # you can use the "save_image" function for this
    # get samples
    #############################################


    # fill in code here  
    #on final epoch, extract filters from model.conv1 and save them
    #as an image. you can use the "save_image" function for this
    #get samples
    if epoch == args.epochs:
        #get samples
        samples = model.conv1.weight.data.cpu()
        #save samples
        save_image(samples, 'prob4_CNN.png', nrow=8, normalize=True, range=(-1,1))
            
    #############################################

    if epoch == args.epochs:
        #create confusion_matrix
        predictions = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions)
        #get targets
        targets = []
        for data, target in test_loader:
            targets.append(target.cpu().numpy())
        targets = np.concatenate(targets)

        cm = confusion_matrix(targets, predictions)

        #plot confusion matrix
        plt.figure(figsize=(10,10))
        plt.imshow(cm, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.title('Confusion Matrix')
        plt.savefig('prob4_confusion_matrix.png')
        #plt.show()
    




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    model = CNN().to(device)

    # ANNOTATION 6
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()
