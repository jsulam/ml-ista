import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import pdb

##################################################

####            MultiLayer ISTA NET           ####

##################################################

class ML_ISTA_NET(nn.Module):
    def __init__(self,m1,m2,m3,T):
        super(ML_ISTA_NET, self).__init__()
        
        self.T = T
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        self.c1 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c2 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c3 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = .1 /np.sqrt(36) * self.W1.data
        self.W2.data = .1 /np.sqrt(m1*36) * self.W2.data
        self.W3.data = .1 /np.sqrt(m2*16) * self.W3.data
        
    def forward(self, x, all_out = False):
        
        # Encoding
        gamma1 = F.relu(self.c1*F.conv2d(x,self.W1, stride = self.strd1) + self.b1)
        gamma2 = F.relu(self.c2*F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(self.c3*F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        
        for _ in  range(self.T):
            
            # backward computation
            gamma2 = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.W2, stride=self.strd2)
            
            # forward computation
            gamma1 = F.relu( (gamma1 - self.c1*F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - self.c2*F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (gamma3 - self.c3*F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
            
        if all_out:    
            # invertibility?
            gamma2 = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.W2, stride=self.strd2)
            x_Rec = F.conv_transpose2d(gamma1,self.W1, stride = self.strd1)
            
            return out, gamma, x_Rec.detach()   
        
        else:
            return out
    

    
##################################################

####          MultiLayer FISTA NET            ####

##################################################

class ML_FISTA_NET(nn.Module):
    def __init__(self,m1,m2,m3,T):
        super(ML_FISTA_NET, self).__init__()
        
        self.T = T
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        self.c1 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c2 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c3 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = .1 /np.sqrt(36) * self.W1.data
        self.W2.data = .1 /np.sqrt(m1*36) * self.W2.data
        self.W3.data = .1 /np.sqrt(m2*16) * self.W3.data
        
    def forward(self, x, all_out = False):
        
        t = 1
        t_prv = t
        
        # Encoding
        gamma1 = F.relu(self.c1*F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       
        gamma2 = F.relu(self.c2*F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(self.c3*F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        gamma3_prv = gamma3
            
        for _ in  range(self.T):
            
            t_prv = t
            t = float((1+np.sqrt(1+4*t_prv**2))/2)  
            
            Z = gamma3 + (t_prv-1)/t * (gamma3 - gamma3_prv)
            gamma3_prv = gamma3
            
            # backward computation
            gamma2 = F.conv_transpose2d(Z,self.W3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.W2, stride=self.strd2)
            
            # forward computation
            gamma1 = F.relu( (gamma1 - self.c1*F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - self.c2*F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (Z - self.c3*F.conv2d( F.conv_transpose2d(Z,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
                    
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        if all_out:    
            # invertibility?
            gamma2 = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.W2, stride=self.strd2)
            x_Rec = F.conv_transpose2d(gamma1,self.W1, stride = self.strd1)
            
            return out, gamma, x_Rec.detach()   
        
        else:
            return out 
    
    
    
    
##################################################

####               ML-LISTA NET               ####

##################################################


class ML_LISTA_NET(nn.Module):
    def __init__(self,m1,m2,m3,T):
        super(ML_LISTA_NET, self).__init__()
        
        self.T = T
        
        # B matrices
        self.B1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.B2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.B3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        
        # W matrices
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        self.strd1 = 2;
        self.strd2 = 2;
        self.strd3 = 1;
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = .1 /np.sqrt(36) * self.W1.data
        self.W2.data = .1 /np.sqrt(m1*36) * self.W2.data
        self.W3.data = .1 /np.sqrt(m2*16) * self.W3.data
        
        self.B1.data = .1 /np.sqrt(36) * self.B1.data
        self.B2.data = .1 /np.sqrt(m1*36) * self.B2.data
        self.B3.data = .1 /np.sqrt(m2*16) * self.B3.data
        
        
    def forward(self, x, all_out = False):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.B1, stride = self.strd1) + self.b1)       # first estimation
        gamma2 = F.relu(F.conv2d(gamma1,self.B2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.B3, stride = self.strd3) + self.b3) 
        
        for _ in  range(self.T): 
            
            gamma2 = F.conv_transpose2d(gamma3,self.B3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.B2, stride=self.strd2)
            
            # forward computation
            gamma1 = F.relu( gamma1 - F.conv2d(F.conv_transpose2d(gamma1,self.W1, stride = self.strd1),self.W1, stride = self.strd1)
                            + F.conv2d( x ,self.B1, stride = self.strd1) + self.b1 )
            gamma2 = F.relu( gamma2 - F.conv2d(F.conv_transpose2d(gamma2,self.W2, stride = self.strd2),self.W2, stride = self.strd2)
                            + F.conv2d( gamma1 ,self.B2, stride = self.strd2) + self.b2 )
            gamma3 = F.relu( gamma3 - F.conv2d(F.conv_transpose2d(gamma3,self.W3, stride = self.strd3),self.W3, stride = self.strd3)
                            + F.conv2d( gamma2 ,self.B3, stride = self.strd3) + self.b3 )
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
        
        if all_out:
            return out,gamma
        else:
            return out
    

        
##################################################

####               Layered BP NET               ####

##################################################


class LBP_NET(nn.Module):
    def __init__(self,m1,m2,m3,T):
        super(LBP_NET, self).__init__()
        
        self.T = T
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        self.c1 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c2 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        self.c3 = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = .1 /np.sqrt(36) * self.W1.data
        self.W2.data = .1 /np.sqrt(m1*36) * self.W2.data
        self.W3.data = .1 /np.sqrt(m2*16) * self.W3.data
        
    def forward(self, x, all_out = False):
        
        
        if self.T == 0:
            # Just a CNN
            gamma1 = F.relu(self.c1*F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       
            gamma2 = F.relu(self.c2*F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
            gamma3 = F.relu(self.c3*F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        
        else:
            # 1st layer
            gamma1 = F.relu(self.c1*F.conv2d(x,self.W1, stride = self.strd1) + self.b1)
            for _ in  range(self.T):
                gamma1 = F.relu( (gamma1 - self.c1*F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)    
                
            # 2nd layer
            gamma2 = F.relu(self.c2*F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
            for _ in  range(self.T):
                gamma2 = F.relu( (gamma2 - self.c2*F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            
            # 3rd layer
            gamma3 = F.relu(self.c3*F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
            for _ in  range(self.T):
                gamma3 = F.relu( (gamma3 - self.c3*F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
            
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
    
        if all_out:    
            # invertibility?
            gamma2 = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1 = F.conv_transpose2d(gamma2,self.W2, stride=self.strd2)
            x_Rec = F.conv_transpose2d(gamma1,self.W1, stride = self.strd1)
            
            return out, gamma, x_Rec.detach()   
        
        else:
            return out  
   




##################################################

####               All Free NET               ####

##################################################



class All_Free(nn.Module):
    def __init__(self,m1,m2,m3):
        super(All_Free, self).__init__()
                
        # Convolutional Filters
        self.W1_1 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_2 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_3 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_4 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_5 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_6 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.W1_7 = nn.Parameter(.1 /np.sqrt(36) * torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        
        self.W2_1 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_2 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_3 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_4 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_5 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_6 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)
        self.W2_7 = nn.Parameter(.1 /np.sqrt(m1*36) * torch.randn(m2,m1,6,6), requires_grad=True)        
        self.strd2 = 2;
        
        self.W3_1 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_2 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_3 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_4 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_5 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_6 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.W3_7 = nn.Parameter(.1 /np.sqrt(m2*16) * torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
                
        # Biases / Thresholds
        self.b1_1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_2 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_3 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_4 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_5 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_6 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b1_7 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        
        self.b2_1 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_3 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_4 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_5 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_6 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b2_7 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        
        self.b3_1 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_2 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_4 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_5 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_6 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        self.b3_7 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        
    def forward(self, x, all_out = False):
        
        # iter 0
        gamma1 = F.relu(F.conv2d(x,self.W1_1, stride = self.strd1) + self.b1_1)
        gamma2 = F.relu(F.conv2d(gamma1,self.W2_1, stride = self.strd2) + self.b2_1) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3_1, stride = self.strd3) + self.b3_1) 

        # iter 1
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_1, stride = self.strd1) - x ,self.W1_2, stride = self.strd1)) + self.b1_2)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_1, stride = self.strd2) - gamma1, self.W2_2, stride = self.strd2)) + self.b2_2) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_1, stride = self.strd3) - gamma2, self.W3_2, stride = self.strd3)) + self.b3_2) 

        # iter 2
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_2, stride = self.strd1) - x ,self.W1_3, stride = self.strd1)) + self.b1_3)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_2, stride = self.strd2) - gamma1, self.W2_3, stride = self.strd2)) + self.b2_3) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_2, stride = self.strd3) - gamma2, self.W3_3, stride = self.strd3)) + self.b3_3) 

        # iter 3
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_3, stride = self.strd1) - x ,self.W1_4, stride = self.strd1)) + self.b1_4)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_3, stride = self.strd2) - gamma1, self.W2_4, stride = self.strd2)) + self.b2_4) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_3, stride = self.strd3) - gamma2, self.W3_4, stride = self.strd3)) + self.b3_4) 

        # iter 4
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_4, stride = self.strd1) - x ,self.W1_5, stride = self.strd1)) + self.b1_5)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_4, stride = self.strd2) - gamma1, self.W2_5, stride = self.strd2)) + self.b2_5) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_4, stride = self.strd3) - gamma2, self.W3_5, stride = self.strd3)) + self.b3_5) 

        # iter 5
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_5, stride = self.strd1) - x ,self.W1_6, stride = self.strd1)) + self.b1_6)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_5, stride = self.strd2) - gamma1, self.W2_6, stride = self.strd2)) + self.b2_6) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_5, stride = self.strd3) - gamma2, self.W3_6, stride = self.strd3)) + self.b3_6) 

        # iter 6
        gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1_6, stride = self.strd1) - x ,self.W1_7, stride = self.strd1)) + self.b1_7)
        gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2_6, stride = self.strd2) - gamma1, self.W2_7, stride = self.strd2)) + self.b2_7) 
        gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3_6, stride = self.strd3) - gamma2, self.W3_7, stride = self.strd3)) + self.b3_7) 
            
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
            
        if all_out:                
            return out, gamma
        
        else:
            return out