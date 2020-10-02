#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT THE REQUIRED PACKAGES

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn

#READ THE DATA
mat = sio.loadmat("./MNIST.mat")

Xtest = mat['input_images']
Ytest = mat['output_labels']


# In[2]:


#DEFINE THE NETOWRK
class Net(nn.Module):
    
    def __init__(self, LN):
        super(Net, self).__init__()
        #number of hidden layers
        numberL = len(LN)
        self.network = nn.Sequential()
        #first layer
        self.network.add_module('net1',nn.Linear(in_features=784, out_features=LN[0]))  
        self.network.add_module('act1',nn.LeakyReLU())   
   
        #hidden layers
        j=2
        for i in range(numberL-1):
            name = 'net'+str(i+2)
            self.network.add_module(name,nn.Linear(in_features=LN[i], out_features=LN[i+1]))
            name = 'act'+str(i+2) 
            self.network.add_module(name,nn.LeakyReLU())
            j += 1

        name = 'net'+str(j)
        self.network.add_module(name,nn.Linear(in_features=LN[-1], out_features=10))

    def forward(self, x):
        #compute up the last layer
        out = self.network(x)
        return out

    def output(self, x):
        #returns the class 
        with torch.no_grad():
            out = self.forward(x)
            out = nn.functional.softmax(out,dim=1)
            _, predicted = torch.max(out, 1)

        return predicted


# In[3]:


#FUNCTION FOR THE ACCURACY
def Accuracy(net,X,Y,Plot=True):
    X = torch.Tensor(X).float().view(-1, X.shape[1])
    Y = torch.LongTensor(Y).squeeze()

    Ypred = net.output(X)
    temp = Ypred-Y
    temp = temp.cpu().numpy()

    #mask the array
    temp[temp>0] = -1
    temp[temp==0] = 1
    temp[temp<0] = 0

    #count the correct
    correct = np.sum(temp)

    return correct/len(Y)


# In[4]:


#LOAD THE NETWORK AND COMPUTE THE ACCURACY ON THE TEST SET
#load the network
#initialize the net
net = Net((480, 132))
#load the state dict previously saved
net_state_dict = torch.load('NetParameters.torch')
      
#update the network parameters
net.load_state_dict(net_state_dict)

#compute the accuracy on the test set
print('The accuracy on the test set is: %.3f' % Accuracy(net,Xtest,Ytest))


# In[ ]:




