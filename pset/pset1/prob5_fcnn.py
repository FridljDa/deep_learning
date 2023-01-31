""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with PyTorch. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
"""

import torch
import torch.nn as nn  # neural network modules 
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fcnn')

torch.manual_seed(42)

# ##################################
# IMPORT DATA
# ##################################

# Download the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# Separate into data and labels
# Reducing training dataset to 1000 points and test dataset to 2000 points in order to create an overfitting model on 
# which to study regularization later

# training data
train_data = mnist_trainset.data.to(dtype=torch.float32)[:1000]
train_data = train_data.reshape(-1, 784)
train_labels = mnist_trainset.targets.to(dtype=torch.long)[:1000]

print(f"train data shape: {train_data.size()}")
print(f"train label shape: {train_labels.size()}")

# testing data
test_data = mnist_testset.data.to(dtype=torch.float32)[:2000]
test_data = test_data.reshape(-1, 784)
test_labels = mnist_testset.targets.to(dtype=torch.long)[:2000]

print(f"test data shape: {test_data.size()}")
print(f"test label shape: {test_labels.size()}")

# Load into torch datasets
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# ##################################
# SET HYPERPARAMETERS
# ##################################

# parameters
learning_rate = 0.01 # Ha ha! This means it will learn really quickly, right?
#TODO Daniel increase epochs
num_epochs = 100 # Training for a long time to see overfitting
batch_size = 128
n_hidden_1 = 64

# network parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# ##################################
# DEFINE THE MODEL 
# ##################################

# Method 1: define a python class, which inherits the rudimentary functionality of a neural network from nn.Module

class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_1=64, p=None):
        super(FCNN, self).__init__()
        
        # As you'd guess, these variables are used to set the number of dimensions coming in and out of the network. We
        # supply them when we initialize the neural network class.
        # Adding them to the class as variables isn't strictly necessary in this case -- but it's good practice to do
        # this book-keeping, should you need to reference the input dim from somewhere else in the class.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_1 = n_hidden_1 # 1st layer number of neurons
        self.p = p # For dropout experiments in part 5.3

        # And here we have the PyTorch magic -- a single call of nn.Linear creates a single fully connected layer with
        # the specified input and output dimensions. All of the parameters are created automatically.
        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, output_dim)
        # TODO 5.1: Create possible extra layers here

        # You can find many other nonlinearities on the PyTorch docs.
        self.nonlin1 = nn.Sigmoid()
        # TODO 5.1: You might try some different activation functions here

    def forward(self, x):
        if self.p is None:
            # When you give your model data (e.g., by running `model(data)`, the data gets passed to this forward 
            # function -- the network's forward pass.

            # You can very easily pass this data into your model's layers, reassign the output to the variable x, and 
            # continue.
            # TODO 5.1: Play with the position of the nonlinearity, and with the number of layers
            x = self.layer1(x)
            x = self.nonlin1(x)
            x = self.layer2(x)

        else:
            # TODO 5.3: Apply dropout to both hidden layers, using a special type of PyTorch layer
            # -- https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
            pass

        return x

# Alternative way of defining a model in pytorch: You can create an equivalent model to FCNN above using nn.Sequential
#
# model2 = nn.Sequential(nn.Linear(num_input, n_hidden_1),
#                        nn.Sigmoid(),
#                        nn.Linear(n_hidden_1, num_classes))

# ##################################
# HELPER FUNCTIONS
# ##################################

def get_accuracy(output, targets):
    """
    calculates accuracy from model output and targets
    """
    output = output.detach()
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / output.size(0) * 100
    return accuracy


def to_one_hot(y, c_dims=10):
    """
    converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def lp_reg(params, p=1):
    """
    compute the Lp regularization, where p is given as a parameter of the function
    """
    total = 0
    for w in params:
        if len(w.shape) > 1: # if this isn't a bias
            total += torch.sum(w**p)
    return total ** (1/p)

# ##################################
# CREATE DATALOADER
# ##################################
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TODO 5.2: Defining loss functions
loss_functions = {
    "CE": torch.nn.CrossEntropyLoss()
}

#plot 
def plot_accuracies_v_epoch(metric_array, experiment_name, plot_training=True, ax=None):

    # You can pass the same axis to the plot function
    # to plot multiple lines on a single figure
    #
    # Ex.
    # fig, ax = plt.subplots()
    # plot_accuracies_v_epoch(metric_array1, experiment_name1, ax=ax)
    # plot_accuracies_v_epoch(metric_array2, experiment_name2, ax=ax)
    # plt.show()
    #
    # This code will produce 1 plot with 2 lines

    if ax is None:
        fig, ax = plt.subplots()

    title = "Training accuracies" if plot_training else "Testing accuracies"
    index = 0 if plot_training else 1

    epochs = np.arange(1, len(metric_array)+1)
    ax.plot(epochs, metric_array[:,index], label=experiment_name)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    if ax is None:
        plt.show()
    else :
        return ax

# ##################################
# MAIN TRAINING FUNCTION
# ##################################

def train(learning_rate = learning_rate, num_epochs=num_epochs, n_hidden_1=n_hidden_1, loss_functions_label = "CE", p=None):
    # HINT: You can pass in arguments to our training function that may be
    # hyperparameters, loss functions, regularization terms etc.
    # Ex.
    # def train(learning_rate=1000000000000000, num_epochs=1000, n_hidden_1=64, ...):

    model = FCNN(num_input, num_classes, n_hidden_1 = n_hidden_1, p=p)

    # TODO 5.2: Choose the loss function
    loss_func = loss_functions[loss_functions_label]

    # TODO 5.1: Choose the optimizer and learning rate
    # TODO 5.3: L2 regularization could be implement here
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Initialize loss list
    metrics = [[0, 0]]

    # Iterate over epochs
    for ep in range(num_epochs):
        model.train()

        # Iterate over batches
        for batch_indx, batch in enumerate(trainloader):

            # This is the code that runs every batch
            # ...

            # unpack batch
            data, labels = batch

            ###################
            # TODO 5.1
            ##################
            # Complete the training loop by feeding the data to the model, comparing the output to the actual labels to
            # compute a loss, backpropogating the loss, and updating the model parameters.
            
            #feed data to model
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # Your code here

            # TODO 5.2: MSE loss requires labels to be one-hot vectors
            # TODO 5.3: Regularization could be implement here


        # And here you might put things that run every epoch
        # ...

        # Compute full train and test accuracies every epoch
        model.eval() # Model will not calculate gradients for this pass, and will disable dropout
        train_ep_pred = model(train_data)
        test_ep_pred = model(test_data)

        train_accuracy = get_accuracy(train_ep_pred, train_labels)
        test_accuracy = get_accuracy(test_ep_pred, test_labels)

        # Save the training and testing accuracies
        metrics.append([train_accuracy, test_accuracy])

        # Print loss every 10 epochs (you can change this frequency)
        if ep % 10 == 0:
            print(f"train acc: {train_accuracy:.2f}\t test acc: {test_accuracy:.2f}\t at epoch: {ep}")
            #add to writer 
            writer.add_scalar("train_accuracy", train_accuracy, ep)
            writer.add_scalar("test_accuracy", test_accuracy, ep)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            #writer.add_figure('predictions vs. actuals',
            #                plot_accuracies_v_epoch(np.array(metrics), f"2 layers, sigmoid, lr = {learning_rate}"),
            #                global_step=ep)

        
    #save the following parameters learning_rate, num_epochs, n_hidden_1, loss_functions_label, p
    writer.add_hparams({"learning_rate": learning_rate, "num_epochs": num_epochs, "n_hidden_1": n_hidden_1, "loss_functions_label": loss_functions_label, "p": p}, {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
    writer.close()
    return np.array(metrics), model

# So using the training function, you would ultimately be left with your metrics (in this case accuracy vs epoch) and 
# your trained model.
# 
# Ex. 
# metric_array, trained_model = train()


if __name__ == '__main__':
    metrics, model = train()
    #plot_accuracies_v_epoch(metrics, f"2 layers, sigmoid, lr = {learning_rate}")

    #fig, ax = plt.subplots()
    #plot_accuracies_v_epoch(metrics, f"train, 2 layers, sigmoid, lr = {learning_rate}", ax=ax)
    #plot_accuracies_v_epoch(metrics, f"test , 2 layers, sigmoid, lr = {learning_rate}", plot_training=False, ax=ax)
    #plt.show()
