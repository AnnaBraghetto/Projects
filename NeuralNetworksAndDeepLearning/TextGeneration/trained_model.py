# -*- coding: utf-8 -*-
import json
import re
import string
import numpy as np
import torch
from torch import nn
import argparse
from pathlib import Path


##############################
##############################
## PARAMETERS
##############################

parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed', type=str, default='the', help='Initial text of the chapter')
parser.add_argument('--length', type=str, default=10, help='Number of words')

##############################
##############################
##############################
#%% NETWORK CLASS
class Network(nn.Module):
    
    def __init__(self, vocab_len, w2v_size, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super(Network, self).__init__()
      
        #vocab_len -> w2v_size
        self.embedding = nn.Embedding(vocab_len, w2v_size)

        # LSTM 
        #w2v_size -> hidden_units
        self.rnn = nn.LSTM(input_size=w2v_size, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        
        # Output
        #hidden_units -> vocab_len
        self.out = nn.Linear(hidden_units, vocab_len)
        
    def forward(self, x, state=None):
        # Embedding
        x = self.embedding(x)
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        output = self.out(x)

        return output, rnn_state
#%% AUXILIARY FUNCTIONS

def Clean(input_text):
    # Lower case
    text = input_text.lower()

    # Remove space after a new line
    text = re.sub('\n[ ]+\n', '\n', text)

    # Substitute cases
    text = re.sub('['+'_'+']', ' ', text)
    text = re.sub('['+'—'+']', ' ', text)
    text = re.sub('['+'-'+']', ' ', text)
    text = re.sub('['+'“'+']', ' ', text)
    text = re.sub('['+'”'+']', ' ', text) 
    text = re.sub('['+'‘'+']', ' ', text) 
    text = re.sub('['+'’'+']', ' ', text) 
    # Keep just points and commas
    text = re.sub('['+','+']', ' '+'commapunct'+' ', text)
    text = re.sub('['+'.'+']', ' '+'pointpunct'+' ', text)
    text = re.sub('['+'!'+']', ' '+'exclapunct'+' ', text)
    text = re.sub('['+'?'+']', ' '+'questpunct'+' ', text)

    return text

def encode_text(dictionary,text):
    i = -1
    for w in text:
        i+=1
        try:
            temp = dictionary[w]
        except:
            text.remove(w)
    encoded = [dictionary[w] for w in text]
    return encoded
def decode_text(dictionary,code):
    text = [dictionary[i] for i in code]
    return text

#%% PREDICTION
def predict(input_seed,n,net,i2w,w2i,w2a):
    seed = Clean(input_seed)
    translator=str.maketrans('','',string.punctuation)
    seed = seed.translate(translator).split() 
    # Evaluation mode
    net.eval() 
    ###  Find initial state of the RNN
    with torch.no_grad():
        # Encode seed
        seed_encoded = encode_text(w2i, seed)
        # To tensor
        seed_tensor = torch.LongTensor(seed_encoded)
        # Add batch axis
        seed_tensor = seed_tensor.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed_tensor)
    # Get the most probable last output index
    next_encoded = next_softmax(net_out[:, -1, :])
    # Print the seed letters
    print(input_seed, end='', flush=True)
    next_decoded = i2w[str(next_encoded)]
    print(w2a[next_decoded], end='', flush=True)

    ### Generate sentences
    tot_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            net_input = torch.LongTensor([next_encoded])
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Get the most probable letter index
            next_encoded = next_softmax(net_out[:, -1, :])
           
            # Decode the letter
            next_decoded = i2w[str(next_encoded)]
            print(w2a[next_decoded], end='', flush=True)
            # Count total letters
            tot_count += 1
            # Break if n words
            if tot_count > n:
                break

def next_softmax(x):
    #returns the word
    with torch.no_grad():
        out = nn.functional.softmax(x,dim=1)
        vocab = np.arange(out.shape[1])
        sampling = out.reshape(-1,).cpu().detach().numpy()
        predicted = np.random.choice(vocab, p=sampling/sum(sampling))
    return predicted.item()
# RUN THE PREDICTION
    
#%% Parse input arguments
args = parser.parse_args()
      
#%% Load encoder and decoder dictionaries
i2w = json.load(open('i2w.json'))
w2i = json.load(open('w2i.json'))
w2a = json.load(open('w2a.json'))
    
#%% Initialize network
net = Network(17246, 100, 128, 2, 0.3)
    
#%% Load network trained parameters
net.load_state_dict(torch.load('net_params.pth', map_location='cpu'))
net.eval() # Evaluation mode (e.g. disable dropout)

#%% Prediction
predict(args.seed,int(args.length),net,i2w,w2i,w2a)