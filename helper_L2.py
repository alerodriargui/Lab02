#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###########################################################################
# Import libraries and set up configuration parameters
###########################################################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
##############################################################################
# Provided functions
##############################################################################


# Test step
def testing_pass(data, target, model, criterion):
    """
    Evaluates a given model on a single batch of data.

    Args:
        data (torch.Tensor): Input data tensor.
        target (torch.Tensor): Target tensor.
        model (torch.nn.Module): PyTorch model to be evaluated.
        criterion (torch.nn.Module): Loss function.

    Returns:
        Tuple containing the loss value and output tensor.
    """
    data, target = data.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        output = model(data)
    loss = criterion(output, target)
    return loss.item(), output


def do_test(model, loaders, criterion):
    """
    Test the performance of the given model on the test set.

    Args:
        model: A PyTorch model.
        loaders: A dictionary with 'test' DataLoader.
        criterion: A PyTorch loss function.

    Returns:
        accuracy: A float representing the accuracy on the test set.
        test_loss: A float representing the average test loss.
        all_outputs: A tensor containing the outputs of the model on the test
        set.
    """

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables
    accumulated_loss = 0.0  # Accumulated loss over the batches
    num_correct_pred = 0  # Number of correct predictions
    num_pred = 0  # Total number of predictions
    accuracy = 0.0  # Accuracy
    all_outputs = torch.tensor([]).to(DEVICE)  # Tensor to hold the outputs

    # Iterate over each batch
    for data, target in loaders['test']:
        # Compute the loss and output of the batch
        batch_loss, output = testing_pass(data, target, model, criterion)

        # Add the batch loss to the accumulated loss. The batch loss are weighted
        # by the number of items
        accumulated_loss += batch_loss * data.size(0)

        # Compute the predicted labels by selecting the indices of the maximum
        # values in the output tensor
        _, pred_labels = torch.max(output, dim=1)

        # Select the actual labels. This is a b-element tensor that will be
        # compared to pred_labels. Notice that true_labels is in the cpu()
        true_labels = target.data.view_as(pred_labels)

        # Compares (item by item) whether the predicted and actual indices are
        # the same
        correct = torch.eq(pred_labels.cpu(), true_labels)

        # Add up the hits
        num_correct_pred += torch.sum(correct).item()
        num_pred += len(target)

        # Concatenate the output of each batch to the all_outputs tensor
        all_outputs = torch.cat((all_outputs, output), dim=0)

    # Results
    test_loss = accumulated_loss / num_pred  # Mean loss per batch
    accuracy = num_correct_pred / num_pred

    return accuracy, test_loss, all_outputs


# %%
def inspect_obj(obj, internal=False):
    """Return the attributes (properties and methods) of an object"""

    """Return a dictionary with three elements. The first element has
    'properties' as key and its value is a list of strings with all the
    properties that the dir() function was able to retrieve.
    The second element of the dictionary, with 'methods' key, is the equivalent
    applied to methods.
    The third element is the union of the previous two, and it's key is
    'attributes'.
    You might want to take a look at the 'inspect' library if you need to dig
    deeper. An example of use would be:
    print(inspect_obj(obj)['properties'])

    Parameters
    ----------
    obj :
        TYPE: object
        DESCRIPTION: It can be any object.
    internal :
        TYPE: bool
        DESCRIPTION: If True it also returns the attributes that start with
            underscore.

    Returns
    -------
    output :
        TYPE: Dictionary of two elements of the type list of strings.
        DESCRIPTION. Dictionary with two elements. The first is
            output['properties'] and the second is output['methods']. They list
            the properties and methods respectively.
    """

    dir_obj = []

    # Loop through attributes found by dir(). This first filter is done because
    # sometimes there are attributes that raise an error when called by
    # getattr() due to they haven't been initialized, or due to they have a
    # special behavior.
    for func in dir(obj):
        try:
            _ = getattr(obj, func)
            dir_obj.append(func)
        except BaseException:
            pass

    # Selection of methods and properties
    if internal:
        method_list = [func for func in dir_obj if callable(getattr(obj,
                                                                    func))]
        property_list = [prop for prop in dir_obj if prop not in method_list]
    else:
        method_list = [func for func in dir_obj if callable(
            getattr(obj, func)) and not func.startswith('_')]
        property_list = [prop for prop in dir_obj if
                         prop not in method_list and not prop.startswith('_')]

    return {'properties': property_list, 'methods': method_list,
            'attributes': sorted(property_list + method_list)}


# Train step
def train_pass(data, target, model, optimizer, criterion):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model.forward(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item(), output


# Validation step
def valid_pass(data, target, model, criterion):
    data, target = data.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        output = model.forward(data)
    loss = criterion(output, target)
    return loss.item(), output


# Saving the model
def trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list,
                 verbose=True):
    custom_dict = {'model_state_dict': model.state_dict(),
                   'opt_state_dict': optimizer.state_dict(),
                   'tr_loss_list': tr_loss_list,
                   'vl_loss_list': vl_loss_list}
    torch.save(custom_dict, filename)
    if verbose:
        print('Checkpoint saved at epoch {}'.format(len(tr_loss_list)))


# Load the model saved with 'trained_save()'
def trained_load(filename, model, optimizer):
    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    checkpoint.pop('model_state_dict')
    checkpoint.pop('opt_state_dict')

    return model, optimizer, checkpoint


# Training function
def train(n_epochs, loaders, model, optimizer, criterion, filename='model.pt',
          checkpoint={}):

    # Get parameters from checkpoint if available
    if bool(checkpoint):
        tr_loss_list = checkpoint['tr_loss_list']
        vl_loss_list = checkpoint['vl_loss_list']
        valid_loss_min = np.min(vl_loss_list)
        trained_epochs = len(tr_loss_list)
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        tr_loss_list = []
        vl_loss_list = []
        valid_loss_min = np.inf
        trained_epochs = 0
        best_state_dict = {}

    # Loop through epochs
    for epoch in range(1 + trained_epochs, n_epochs + trained_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss, valid_loss = 0.0, 0.0

        # Training
        for data, target in loaders['train']:
            train_loss += train_pass(data, target, model, optimizer, criterion)[0]
        # Training losses log
        tr_loss_list.append(train_loss / len(loaders['train']))

        # Validation
        model.eval()
        for data, target in loaders['valid']:
            valid_loss += valid_pass(data, target, model, criterion)[0]
        # Validation losses log
        vl_loss_list.append(valid_loss / len(loaders['valid']))

        # Results
        end_time = time.time()
        print('Epoch: {} \tTraining loss: {:.5f} \tValidation loss: {:.5f}\
            \t Time: {:.1f} s'.format(epoch, tr_loss_list[-1],
                                      vl_loss_list[-1], end_time - start_time))

        # Saving best model
        if vl_loss_list[-1] < valid_loss_min:
            best_state_dict = copy.deepcopy(model.state_dict())
            trained_save(filename, model, optimizer,
                         tr_loss_list, vl_loss_list)
            valid_loss_min = vl_loss_list[-1]

    # The best model is returned and the training data are written before
    # exiting
    model.load_state_dict(best_state_dict)
    trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list, False)

    return model, (tr_loss_list, vl_loss_list)


# Traing plot
def plot_checkpoint(checkpoint):
    x = range(1, 1 + len(checkpoint['tr_loss_list']))
    tr_losses = checkpoint['tr_loss_list']
    vl_losses = checkpoint['vl_loss_list']
    tr_max, tr_min = np.max(tr_losses), np.min(tr_losses)
    epoch_min = 1 + np.argmin(vl_losses)
    val_min = np.min(vl_losses)

    plt.plot(x, tr_losses, label='training loss')
    plt.plot(x, vl_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Losses during training")
    plt.legend()
    plt.annotate('valid min: {:.4f}'.format(val_min),
                 xy=(epoch_min,
                     val_min),
                 xytext=(round(0.75 * len(tr_losses)),
                         3 * (tr_max - tr_min) / 4 + tr_min),
                 arrowprops=dict(facecolor='black',
                                 shrink=0.05),
                 )
    plt.xlim(0, len(tr_losses))
    plt.show()


# %%
##############################################################################
# Test functions
##############################################################################


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Dense layers
        # Three dense layers are defined. The input of the first hidden layer
        # will have as many units as pixels in the image. The output of the
        # last layer will have as many units as classes we want to identify, in
        # this case 10 digits
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=10, bias=True)

        # Activation layers
        self.relu = nn.ReLU()

        # Dropout layers.
        # The parameter p is the probability of each unit to be turned off in
        # the current epoch. We'll see an example shortly. The dropout is now
        # set to disabled.
        self.dropout = nn.Dropout(p=0.0)

    # Definition of forward pass method
    def forward(self, x):
        # The inputs will propagate forward through all the defined layers. The
        # behavior is specified by each function.
        x = x.view(-1, 28 * 28)  # in: b x 28 x 28  out: b x 784
        x = self.relu(self.fc1(x))  # in: b x 784  out: b x 256
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # in: b x 256  out: b x 64
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  # in: b x 64  out: b x 10
        # Notice how in this case there is no output activation that converts
        # the scores into probabilities.

        return x

