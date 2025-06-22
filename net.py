import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from core.util import fit_standardizer
from sklearn import preprocessing
import os
import numpy as np
from sys import exit

class MainNet(nn.Module):
    def __init__(self, net_params):
        super(MainNet, self).__init__()
        self.net_params = net_params
        self.opt_parameters_FC = []
        self.opt_parameters_conv = []
        self.opt_parameters_A = []

    def loss(self, output, labels):
        """ Creates the loss function to train the network
        Args:
        output: output of the network
        labels: labels

        returns:
        total_loss: total loss including MSE and regularization error
        
        """
        criterion = nn.MSELoss()
        total_loss = criterion(output, labels)

        if 'l2_reg' in self.net_params and self.net_params['l2_reg'] > 0:
            l2_reg = self.net_params['l2_reg']
            l2_loss = l2_reg*self.get_l2_norm_()
        else:
            l2_loss = 0.0

        if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:
            l1_reg = self.net_params['l1_reg']
            l1_loss = l1_reg*self.get_l1_norm_()
        else:
            l1_loss = 0.0

        return total_loss + l2_loss + l1_loss


    def construct_FC_net(self):
        """
        Construct the fully connected network
        """
        input_dim = self.net_params['input_dim']
        FC_layers = self.net_params['FC_layers']

        self.FC_in = nn.Linear(input_dim, FC_layers[0])
        self.opt_parameters_FC.append(self.FC_in.weight)
        self.opt_parameters_FC.append(self.FC_in.bias)

        self.FC_hidden = nn.ModuleList()

        for i in range(len(FC_layers)-1):
            self.FC_hidden.append(nn.Linear(FC_layers[i], FC_layers[i+1]))
            self.opt_parameters_FC.append(self.FC_hidden[-1].weight)
            self.opt_parameters_FC.append(self.FC_hidden[-1].bias)

    def construct_Conv_net(self):
        """
        Construct a convolutional network
        """
        Conv_layers_RELU = self.net_params['Conv_layers_RELU']
        Conv_layers = self.net_params['Conv_layers']

        self.Conv_RELU = nn.Conv2d(Conv_layers_RELU[0], Conv_layers_RELU[1], Conv_layers_RELU[2])
        self.Conv_last = nn.Conv2d(Conv_layers[0], Conv_layers[1], Conv_layers[2])

        self.opt_parameters_conv.append(self.Conv_RELU.weight)
        self.opt_parameters_conv.append(self.Conv_RELU.bias)

        self.opt_parameters_conv.append(self.Conv_last.weight)
        self.opt_parameters_conv.append(self.Conv_last.bias)

    def construct_A_net(self):
        """
        Constructs a fully connected final linear layer
        """
        num_output = self.net_params['output_dim']

        self.A_layer = nn.Linear(self.net_params['dim_layer_a'], 1, bias = False)
        self.opt_parameters_A.append(self.A_layer.weight)

    def set_activation_fn(self):
        """
        Selects the nonlinear activation function
        """
        activation_FC = self.net_params['activation_FC']
        activation_Conv = self.net_params['activation_Conv']

        if activation_FC == 'relu':
            self.activation_FC = F.relu

        elif activation_FC == 'tanh':
            self.activation_FC = torch.tanh

        else:
            exit("Exit : invalid activation function")

        if activation_Conv == 'relu':
            self.activation_Conv = F.relu

        elif activation_Conv == 'tanh':
            self.activation_Conv = torch.tanh

        else:
            exit("Exit : invalid activation function")

    def forward(self, x_in, Train = True):
        """
        Forward pass through the network
        
        Args:
            x_in: Input tensor of shape (batch_size, input_size)
            
        Returns:
            outputs: Decoded real-valued outputs of shape (batch_size, output_size)
        """
        m = self.net_params['output_dim']
        n = self.net_params['num_joints']

        self.set_activation_fn()
        x = self.activation_FC(self.FC_in(x_in[:,:2*n]))

        Conv_layers_RELU = self.net_params['Conv_layers_RELU']
        Conv_layers = self.net_params['Conv_layers']

        batch_norm1 = nn.BatchNorm2d(Conv_layers_RELU[1])
        batch_norm2 = nn.BatchNorm2d(Conv_layers[1])

        max_pool1 = nn.MaxPool2d(Conv_layers_RELU[2][0])
        max_pool2 = nn.MaxPool2d(Conv_layers[2][0])

        for layer in self.FC_hidden:
            x = self.activation_FC(layer(x))

        #convolutional operation
        x_conv = torch.cat((x_in[:,n:2*n].unsqueeze(1), x.unsqueeze(1),x_in[:,2*n:].unsqueeze(1)), 1)
        out1 = max_pool1(self.activation_Conv(self.Conv_RELU(x_conv.unsqueeze(3))))
        out2 = max_pool2(batch_norm2(self.Conv_last(batch_norm1(out1))))
        # out2 = self.Conv_last(out1)

        # for i in range(m):
        out_a = self.A_layer(out2[:,:,0,:].squeeze()).squeeze()
        out_b = self.A_layer(out2[:,:,1,:].squeeze()).squeeze()
        out_c = self.A_layer(out2[:,:,2,:].squeeze()).squeeze()

        if m==3:
            c = torch.vstack((out_a,out_b,out_c))
        elif m==2:
            c = torch.vstack((out_a,out_b))

        if Train:
            return c.T
        else:
            return c.T.detach().numpy()

    def A_weight(self):
        """ Returns the weight of the final linear layer"""
        return self.A_layer.weight.T.detach().numpy()

    def Y_out(self, input):
        """
        Returns the output of the network except the final linear layer
        """
        m = self.net_params['output_dim']
        n = self.net_params['num_joints']

        self.set_activation_fn()
        x = self.activation_FC(self.FC_in(input[:,:2*n]))

        Conv_layers_RELU = self.net_params['Conv_layers_RELU']
        Conv_layers = self.net_params['Conv_layers']

        batch_norm1 = nn.BatchNorm2d(Conv_layers_RELU[1])
        batch_norm2 = nn.BatchNorm2d(Conv_layers[1])

        max_pool1 = nn.MaxPool2d(Conv_layers_RELU[2][0])
        max_pool2 = nn.MaxPool2d(Conv_layers[2][0])

        for layer in self.FC_hidden:
            x = self.activation_FC(layer(x))

        a = input[:,2*n:3*n].unsqueeze(1)
        b = x.unsqueeze(1)
        c = input[:,3*n:].unsqueeze(1)
        #convolutional operation
        x_conv = torch.cat((input[:,2*n:3*n].unsqueeze(1), x.unsqueeze(1),input[:,3*n:].unsqueeze(1)), 1)
        out1 = max_pool1(self.activation_Conv(self.Conv_RELU(x_conv.unsqueeze(3))))
        out2 = max_pool2(batch_norm2(self.Conv_last(batch_norm1(out1))))

        return out2.detach().numpy()

    def preprocess_data(self, data, standardized):
        """Standardizes the input dataset"""
        if standardizer is None:
            data_scaled = data
        else:
            data_scaled = standardizer.transform(data)
        return data_scaled


    def process_data(self, data_x, data_u):
        """Flattens the input dataset"""
        n = self.net_params['num_joints']
        m = self.net_params['output_dim']
        N = self.net_params['input_dim']

        order = 'F'
        n_data_pts = data_x.shape[0] * (data_x.shape[1])
        x_flat = data_x.T.reshape((3*n, n_data_pts), order = order)
        u_flat = data_u.T.reshape((m, n_data_pts), order = order)

        return x_flat, u_flat

    def standardize_data(self, data):
        data_std = fit_standardizer(data, preprocessing.StandardScaler(with_mean = True))
        data_scaled = self.preprocess_data(data, data_std)

        return data_scaled.T


    def get_l2_norm_(self):
        """Computes the l2 norm of the weights of the network"""
        return torch.norm(self.FC_in.weight.view(-1), p = 2) + torch.norm(self.FC_in.bias.view(-1), p = 2) +\
               torch.norm(self.FC_hidden[0].weight.view(-1), p = 2) + torch.norm(self.FC_hidden[0].bias.view(-1), p = 2) + \
               torch.norm(self.FC_hidden[1].weight.view(-1), p = 2) + torch.norm(self.FC_hidden[1].bias.view(-1), p = 2) + \
               torch.norm(self.Conv_RELU.weight.view(-1), p = 2) +  torch.norm(self.Conv_RELU.bias.view(-1), p = 2) + \
               torch.norm(self.Conv_last.weight.view(-1), p = 2) + torch.norm(self.Conv_last.bias.view(-1), p = 2) + \
               torch.norm(self.A_layer.weight.view(-1), p = 2)

    def get_l1_norm_(self):
        """Computes the l1 norm of the weights of the network"""
        return torch.norm(self.FC_in.weight.view(-1), p = 1) + torch.norm(self.FC_in.bias.view(-1), p = 1) + \
               torch.norm(self.FC_hidden[0].weight.view(-1), p = 1) + torch.norm(self.FC_hidden[0].bias.view(-1), p = 1) + \
               torch.norm(self.FC_hidden[1].weight.view(-1), p = 1) + torch.norm(self.FC_hidden[1].bias.view(-1), p = 1) + \
               torch.norm(self.Conv_RELU.weight.view(-1), p = 1) +  torch.norm(self.Conv_RELU.bias.view(-1), p = 1) + \
               torch.norm(self.Conv_last.weight.view(-1), p = 1) + torch.norm(self.Conv_last.bias.view(-1), p = 1) + \
               torch.norm(self.A_layer.weight.view(-1), p = 1)


#create a class to train neural network
class TrainNet:
    def __init__(self, net):
        self.net = net

    def model_pipeline(self, xs_train, us_train, xs_val, us_val, print_epoch = True):
        """
        Prepares the dataset and network for training
        """
        #construct the network
        self.net.construct_FC_net()
        self.net.construct_Conv_net()
        self.net.construct_A_net()

        self.set_optimizer_()

        X_train, Y_train = self.net.process_data(xs_train, us_train)
        X_val, Y_val = self.net.process_data(xs_val, us_val)

        X_train_std = self.net.standardize_data(X_train.T)
        X_val_std = self.net.standardize_data(X_val.T)

        # Y_train_std = self.net.standardize_data(Y_train.T)
        # Y_val_std = self.net.standardize_data(Y_val.T)

        Y_train_std = Y_train
        Y_val_std = Y_val

        X_train_tensor, Y_train_tensor = torch.from_numpy(X_train_std).float(), torch.from_numpy(Y_train_std).float()
        X_val_tensor, Y_val_tensor = torch.from_numpy(X_val_std).float(), torch.from_numpy(Y_val_std).float()

        dataset_train = torch.utils.data.TensorDataset(X_train_tensor.T, Y_train_tensor.T)
        dataset_val = torch.utils.data.TensorDataset(X_val_tensor.T, Y_val_tensor.T)

        self.train_model(dataset_train, dataset_val, print_epoch = print_epoch)


    def train_model(self, dataset_train, dataset_val, print_epoch):
        """
        Trains the network
        """
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size = self.net.net_params['batch_size'], shuffle = True)
        valloader = torch.utils.data.DataLoader(dataset_val, batch_size = self.net.net_params['batch_size'])

        val_loss_prev = np.inf
        self.train_loss_hist = []
        self.val_loss_hist = []

        for epoch in range(self.net.net_params['epochs']):
            running_loss = 0.0
            epoch_steps = 0

            for data in trainloader:
                inputs, labels = data
                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.net.loss(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.detach()
                epoch_steps +=1

            val_loss = 0.0
            val_steps = 0

            for data in valloader:
                with torch.no_grad():
                    inputs, labels = data
                    output = self.net(inputs)
                    loss = self.net.loss(output, labels)

                    val_loss += float(loss.detach())
                    val_steps += 1

            #print epoch loss
            self.train_loss_hist.append((running_loss/epoch_steps))
            self.val_loss_hist.append((val_loss/val_steps))

            if print_epoch:
                print('Epoch %3d: train loss: %.10f, validation loss: %.10f' %(epoch+1, self.train_loss_hist[-1], self.val_loss_hist[-1]))
        print('Finished Training')

    def plot_learning_curve(self):
        """Plots the loss convergence with epochs"""
        import matplotlib.pyplot as plt

        train_loss = np.array(self.train_loss_hist)  # [[total loss, prediction loss, bilinearity loss]]
        val_loss = np.array(self.val_loss_hist)  # [[total loss, prediction loss, bilinearity loss]]
        iter = np.arange(train_loss.shape[0])
        titles = ['Total loss']
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.plot(iter, train_loss[:, 0], label='training')
        plt.plot(iter, val_loss[:, 0], '--', label='validation')
        plt.title(titles[0])
        plt.legend()
        plt.show()

    
    def set_optimizer_(self):
        """ Selects the optimizer """
        if self.net.net_params['optimizer'] == 'adam':
            lr = self.net.net_params['lr']
            weight_decay = self.net.net_params['l2_reg']
            self.optimizer = optim.Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)


    def model_pipeline_adapt(self, x_adapt, u_adapt, adapt_params, print_epoch = True):
        """ Prepares the dataset for adaptation """
        #construct the network

        self.adapt_params = adapt_params
        X_adapt_std = self.net.standardize_data(x_adapt.T)

        Y_adapt_std = u_adapt
        X_adapt_tensor, Y_adapt_tensor = torch.from_numpy(X_adapt_std.T).float(), torch.from_numpy(Y_adapt_std.T).float()

        dataset_adapt = torch.utils.data.TensorDataset(X_adapt_tensor.T, Y_adapt_tensor.T)
        self.train_model_adapt(dataset_adapt, print_epoch = print_epoch)


    def train_model_adapt(self, dataset_adapt, print_epoch):
        """Trains the adaptation model"""
        trainloader = torch.utils.data.DataLoader(dataset_adapt, batch_size = self.adapt_params['batch_size'], shuffle = False)

        self.adapt_loss_hist = []

        for epoch in range(self.adapt_params['epochs']):
            running_loss = 0.0
            epoch_steps = 0

            for data in trainloader:
                inputs, labels = data
                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.net.loss(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.detach()
                epoch_steps +=1

            #print epoch loss
            self.adapt_loss_hist.append((running_loss/epoch_steps))

            if print_epoch:
                print('Epoch %3d: adapt loss: %.10f,' %(epoch+1, self.adapt_loss_hist[-1]))
        print('Finished Training')
