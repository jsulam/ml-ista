import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


##################################################

####            MultiLayer ISTA NET           ####

##################################################

class ML_ISTA_NET(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_ISTA_NET, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 1/np.sqrt(36) * self.W1.data
        self.W2.data = 1/np.sqrt(36*m1) * self.W2.data
        self.W3.data = 1/np.sqrt(16*m2) * self.W3.data
        
    def forward(self, x,T=0,RHO=1):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       # first estimation
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
    
    
    

    
    
##################################################

####          MultiLayer FISTA NET            ####

##################################################

class ML_FISTA_NET(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_FISTA_NET, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 1/np.sqrt(36) * self.W1.data
        self.W2.data = 1/np.sqrt(36*m1) * self.W2.data
        self.W3.data = 1/np.sqrt(16*m2) * self.W3.data
        
    def forward(self, x,T=0,RHO=1):
        
        t = 1
        t_prv = t
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        gamma3_prv = gamma3
            
        for _ in  range(T):
            
            t_prv = t
            t = float((1+np.sqrt(1+4*t_prv**2))/2)  
            
            Z = gamma3 + (t_prv-1)/t * (gamma3 - gamma3_prv)
            gamma3_prv = gamma3
            
            # backward computation
            gamma2_ml = F.conv_transpose2d(Z,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (Z - F.conv2d( F.conv_transpose2d(Z,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
                    
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
    
    
    